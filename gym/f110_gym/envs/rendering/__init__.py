from f110_gym.envs.rendering.renderer import RenderSpec, EnvRenderer


def make_renderer(render_spec: RenderSpec = None) -> EnvRenderer:
    render_spec = render_spec or RenderSpec()
    if render_spec.render_mode == "human_fast":
        from f110_gym.envs.rendering.rendering_pyglet import PygletEnvRenderer
        renderer = PygletEnvRenderer(render_spec=render_spec)
    elif render_spec.render_mode in ["human", "rgb_array"]:
        from f110_gym.envs.rendering.rendering_pygame import PygameEnvRenderer
        renderer = PygameEnvRenderer(render_spec=render_spec)
    else:
        renderer = None
    return renderer
