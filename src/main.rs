use anyhow::Context;
use std::sync::Arc;
use vulkano::{
    app_info_from_cargo_toml,
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer},
    device::{Device, DeviceExtensions, DeviceOwned, Features, Queue},
    format::Format,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract},
    image::{
        swapchain::SwapchainImage, AttachmentImage, ImageAccess, ImageCreationError, ImageUsage,
    },
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        Instance, InstanceExtensions, PhysicalDevice, QueueFamily,
    },
    single_pass_renderpass,
    swapchain::{
        AcquireError, Capabilities, ColorSpace, CompositeAlpha, FullscreenExclusive, PresentMode,
        Surface, Swapchain, SwapchainAcquireFuture,
    },
    sync::GpuFuture,
    sync::SharingMode,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::Size,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::desktop::EventLoopExtDesktop,
    window::{Window, WindowBuilder},
};

fn main() -> anyhow::Result<()> {
    let mut event_loop = EventLoop::new();

    // Load the Vulkan library
    vulkano::instance::loader::auto_loader().context("Couldn't load the Vulkan library")?;

    // Create Vulkan instance
    let instance = create_instance().context("Couldn't create Vulkan instance")?;

    let surface = WindowBuilder::new()
        .with_min_inner_size(Size::Physical([320, 240].into()))
        .with_inner_size(Size::Physical([800, 600].into()))
        .with_title("Ferret")
        .build_vk_surface(&event_loop, instance.clone())
        .context("Couldn't create Vulkan rendering window")?;

    // Setup debug callback for validation layers
    #[cfg(debug_assertions)]
    let _debug_callback = DebugCallback::new(
        &instance,
        MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        },
        MessageType::all(),
        |ref message| {
            println!("{}: {}", message.layer_prefix, message.description);
        },
    )
    .ok();

    #[cfg(not(debug_assertions))]
    let debug_callback = None;

    // Create Vulkan device
    let (device, graphics_queue) =
        create_device(&instance, &surface).context("Couldn't create Vulkan device")?;

    // Create render target
    let window_size = surface.window().inner_size().into();
    let mut target = RenderTarget::new(surface.clone(), device.clone(), window_size)
        .context("Couldn't create render target")?;

    // Create attachments
    let (mut colour_attachment, mut depth_attachment) = create_attachments(&device, window_size)?;

    // Create command buffers for presenting
    let mut present_commands =
        create_command_buffers(&device, &graphics_queue, &target, &colour_attachment)
            .context("Couldn't create present command buffers")?;

    // Create render pass
    let render_pass: Arc<dyn RenderPassAbstract + Send + Sync> = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: colour_attachment.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: depth_attachment.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .context("Couldn't create render pass")?,
    );

    // Create framebuffer
    let mut framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(colour_attachment.clone())?
            .add(depth_attachment.clone())?
            .build()
            .context("Couldn't create framebuffers")?,
    );

    let mut should_quit = false;

    while !should_quit {
        event_loop.run_return(|event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    should_quit = true;
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(_) => {
                    match recreate(
                        &device,
                        &graphics_queue,
                        &render_pass,
                        &surface,
                        &mut target,
                    ) {
                        Ok(recreated) => {
                            target = recreated.0;
                            colour_attachment = recreated.1;
                            depth_attachment = recreated.2;
                            present_commands = recreated.3;
                            framebuffer = recreated.4;
                        }
                        Err(err) => println!("Error recreating swapchain: {}", err),
                    }
                }
                _ => {}
            },
            Event::RedrawEventsCleared => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        });

        let clear_value = vec![[0.0, 0.0, 1.0, 1.0].into(), 1.0.into()];

        let mut draw_commands = AutoCommandBufferBuilder::primary_one_time_submit(
            target.device().clone(),
            graphics_queue.family(),
        )?;

        draw_commands
            .begin_render_pass(framebuffer.clone(), false, clear_value)
            .context("Couldn't begin render pass")?
            .end_render_pass()
            .context("Couldn't end render pass")?;

        let draw_future = draw_commands
            .build()?
            .execute(graphics_queue.clone())
            .context("Couldn't execute draw commands")?;

        // Acquire swapchain image
        let (image_num, swapchain_future) = match target.acquire_next_image() {
            Ok((_, true, _)) | Err(AcquireError::OutOfDate) => {
                let recreated = recreate(
                    &device,
                    &graphics_queue,
                    &render_pass,
                    &surface,
                    &mut target,
                )?;
                target = recreated.0;
                colour_attachment = recreated.1;
                depth_attachment = recreated.2;
                present_commands = recreated.3;
                framebuffer = recreated.4;

                match target.acquire_next_image() {
                    Ok((image_num, false, future)) => (image_num, future),
                    Ok((_, true, _)) => {
                        anyhow::bail!("Swapchain out of date even after recreating")
                    }
                    Err(x) => Err(x).context("Couldn't acquire swapchain framebuffer")?,
                }
            }
            Ok((image_num, false, future)) => (image_num, future),
            Err(x) => Err(x).context("Couldn't acquire swapchain framebuffer")?,
        };

        let present_commands2 =
            create_command_buffers(&device, &graphics_queue, &target, &colour_attachment)?;

        // Present
        draw_future
            .join(swapchain_future)
            .then_execute(graphics_queue.clone(), present_commands[image_num].clone())
            .context("Couldn't execute present command")?
            .then_swapchain_present(
                graphics_queue.clone(),
                target.swapchain().clone(),
                image_num,
            )
            .then_signal_fence_and_flush()
            .context("Couldn't flush command buffer")?
            .wait(None)
            .context("Couldn't flush command buffer")?;
    }

    Ok(())
}

fn create_instance() -> anyhow::Result<Arc<Instance>> {
    let mut instance_extensions = vulkano_win::required_extensions();
    let supported_extensions = InstanceExtensions::supported_by_core().unwrap();

    let mut layers = Vec::new();

    #[cfg(debug_assertions)]
    {
        if supported_extensions.ext_debug_utils {
            instance_extensions.ext_debug_utils = true;

            let available_layers: Vec<_> = vulkano::instance::layers_list()?.collect();

            for to_enable in [
                "VK_LAYER_LUNARG_standard_validation",
                "VK_LAYER_LUNARG_monitor",
            ]
            .iter()
            {
                if available_layers.iter().any(|l| l.name() == *to_enable) {
                    layers.push(*to_enable);
                }
            }

            println!(
                "EXT_debug_utils is available, enabled Vulkan validation layers: {}",
                layers.join(", ")
            );
        } else {
            println!("EXT_debug_utils not available, Vulkan validation layers disabled");
        }
    }

    let instance = Instance::new(
        Some(&app_info_from_cargo_toml!()),
        &instance_extensions,
        layers,
    )?;

    Ok(instance)
}

fn find_suitable_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: &Surface<Window>,
) -> anyhow::Result<Option<(PhysicalDevice<'a>, QueueFamily<'a>)>> {
    for physical_device in PhysicalDevice::enumerate(&instance) {
        let family = {
            let mut val = None;

            for family in physical_device.queue_families() {
                if family.supports_graphics() && surface.is_supported(family)? {
                    val = Some(family);
                    break;
                }
            }

            val
        };

        if family.is_none() {
            continue;
        }

        let supported_extensions = DeviceExtensions::supported_by_device(physical_device);

        if !supported_extensions.khr_swapchain {
            continue;
        }

        let capabilities = surface.capabilities(physical_device)?;

        if capabilities.supported_formats.is_empty()
            || capabilities.present_modes.iter().count() == 0
        {
            continue;
        }

        return Ok(Some((physical_device, family.unwrap())));
    }

    Ok(None)
}

fn create_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
) -> anyhow::Result<(Arc<Device>, Arc<Queue>)> {
    // Select physical device
    let (physical_device, family) = find_suitable_physical_device(&instance, &surface)?
        .context("No suitable physical device found")?;

    let features = Features::none();
    let extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (device, mut queues) =
        Device::new(physical_device, &features, &extensions, vec![(family, 1.0)])?;

    Ok((device, queues.next().unwrap()))
}

pub struct RenderTarget {
    images: Vec<Arc<SwapchainImage<Window>>>,
    swapchain: Arc<Swapchain<Window>>,
}

impl RenderTarget {
    pub fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        dimensions: [u32; 2],
    ) -> anyhow::Result<RenderTarget> {
        let capabilities = surface.capabilities(device.physical_device())?;
        let surface_format =
            choose_format(&capabilities).context("No suitable swapchain format found")?;
        let present_mode = [PresentMode::Mailbox, PresentMode::Fifo]
            .iter()
            .copied()
            .find(|mode| capabilities.present_modes.supports(*mode))
            .unwrap();

        let image_count = u32::min(
            capabilities.min_image_count + 1,
            capabilities.max_image_count.unwrap_or(std::u32::MAX),
        );

        let image_usage = ImageUsage {
            color_attachment: true,
            transfer_source: true,
            transfer_destination: true,
            ..ImageUsage::none()
        };

        // Create swapchain and images
        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface,
            image_count,
            surface_format,
            capabilities.current_extent.unwrap_or(dimensions),
            1,
            image_usage,
            SharingMode::Exclusive,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .context("Couldn't create swapchain")?;

        Ok(RenderTarget { images, swapchain })
    }

    pub fn acquire_next_image(
        &self,
    ) -> Result<(usize, bool, SwapchainAcquireFuture<Window>), AcquireError> {
        vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None)
    }

    pub fn device(&self) -> &Arc<Device> {
        self.swapchain.device()
    }

    pub fn recreate(&mut self, dimensions: [u32; 2]) -> anyhow::Result<RenderTarget> {
        let capabilities = self
            .swapchain()
            .surface()
            .capabilities(self.swapchain.device().physical_device())?;
        let surface_format =
            choose_format(&capabilities).context("No suitable swapchain format found")?;

        let image_usage = ImageUsage {
            color_attachment: true,
            transfer_source: true,
            transfer_destination: true,
            ..ImageUsage::none()
        };

        let (swapchain, images) = Swapchain::with_old_swapchain(
            self.swapchain.device().clone(),
            self.swapchain.surface().clone(),
            self.swapchain.num_images(),
            surface_format,
            capabilities.current_extent.unwrap_or(dimensions),
            1,
            image_usage,
            SharingMode::Exclusive,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            self.swapchain.present_mode(),
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
            self.swapchain.clone(),
        )
        .context("Couldn't create swapchain")?;

        Ok(RenderTarget { images, swapchain })
    }

    pub fn images(&self) -> &Vec<Arc<SwapchainImage<Window>>> {
        &self.images
    }

    pub fn swapchain(&self) -> &Arc<Swapchain<Window>> {
        &self.swapchain
    }
}

fn choose_format(capabilities: &Capabilities) -> Option<Format> {
    let srgb_formats = capabilities
        .supported_formats
        .iter()
        .filter_map(|f| {
            if f.1 == ColorSpace::SrgbNonLinear {
                Some(f.0)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let allowed_formats = [
        Format::B8G8R8A8Unorm,
        Format::R8G8B8A8Unorm,
        Format::A8B8G8R8UnormPack32,
    ];

    allowed_formats
        .iter()
        .cloned()
        .find(|f| srgb_formats.iter().any(|g| g == f))
}

fn create_attachments(
    device: &Arc<Device>,
    dimensions: [u32; 2],
) -> anyhow::Result<(Arc<AttachmentImage>, Arc<AttachmentImage>)> {
    // Create colour attachment
    let colour_attachment = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        Format::R8G8B8A8Unorm,
        ImageUsage {
            color_attachment: true,
            transfer_source: true,
            ..ImageUsage::none()
        },
    )
    .context("Couldn't create colour attachment")?;

    // Create depth attachment
    const DEPTH_FORMATS: [Format; 3] = [
        Format::D32Sfloat,
        Format::X8_D24UnormPack32,
        Format::D16Unorm,
    ];

    let depth_attachment = DEPTH_FORMATS
        .iter()
        .cloned()
        .filter_map(|format| {
            let res = AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                format,
                ImageUsage {
                    depth_stencil_attachment: true,
                    transient_attachment: true,
                    ..ImageUsage::none()
                },
            );
            match res {
                Err(ImageCreationError::FormatNotSupported) => None,
                other => Some(other.context("Couldn't create depth attachment")),
            }
        })
        .next()
        .context("No supported depth buffer format found")??;

    Ok((colour_attachment, depth_attachment))
}

fn create_command_buffers(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    target: &RenderTarget,
    colour_attachment: &Arc<AttachmentImage>,
) -> anyhow::Result<Vec<Arc<AutoCommandBuffer>>> {
    let [width, height, depth] = colour_attachment.dimensions().width_height_depth();

    target
        .images()
        .iter()
        .map(|target_image| {
            let mut builder = AutoCommandBufferBuilder::primary(device.clone(), queue.family())?;
            builder.blit_image(
                colour_attachment.clone(),
                [0, 0, 0],
                [width as i32, height as i32, depth as i32],
                0,
                0,
                target_image.clone(),
                [0, 0, 0],
                [width as i32, height as i32, depth as i32],
                0,
                0,
                1,
                vulkano::sampler::Filter::Nearest,
            )?;
            Ok(Arc::new(builder.build()?))
        })
        .collect()
}

fn recreate(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    surface: &Arc<Surface<Window>>,
    target: &mut RenderTarget,
) -> anyhow::Result<(
    RenderTarget,
    Arc<AttachmentImage>,
    Arc<AttachmentImage>,
    Vec<Arc<AutoCommandBuffer>>,
    Arc<dyn FramebufferAbstract + Send + Sync>,
)> {
    let window_size = surface.window().inner_size().into();
    let target = target
        .recreate(window_size)
        .context("Couldn't recreate render target")?;

    // Create attachments
    let (colour_attachment, depth_attachment) = create_attachments(&device, window_size)?;

    // Create command buffers for presenting
    let present_commands = create_command_buffers(device, queue, &target, &colour_attachment)?;

    // Create framebuffer
    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(colour_attachment.clone())?
            .add(depth_attachment.clone())?
            .build()
            .context("Couldn't create framebuffers")?,
    );

    Ok((
        target,
        colour_attachment,
        depth_attachment,
        present_commands,
        framebuffer,
    ))
}
