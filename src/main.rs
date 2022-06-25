use macroquad::prelude::*;

#[macroquad::main("Fractal")]
async fn main() {
    let render_target = {
        let size = 1000;
        let sizef = size as f32;
        let render_target = render_target(size, size);
        render_target.texture.set_filter(FilterMode::Nearest);

        set_camera(&Camera2D {
            zoom: vec2(2. / sizef, 2. / sizef),
            target: vec2(sizef / 2., sizef / 2.),
            render_target: Some(render_target),
            ..Default::default()
        });

        clear_background(LIGHTGRAY);

        let count = 50;
        for i in (0..count).map(|i| sizef/count as f32 * i as f32) {
            draw_line(0., i, sizef, i, 2.0, BLUE);
            draw_line(i, 0., i, sizef, 2.0, BLUE);
        }

        set_default_camera();

        render_target
    };

    let material = load_material(
        VERTEX_SHADER,
        FRAGMENT_SHADER,
        MaterialParams {
            uniforms: vec![
                ("_resolution".to_owned(), UniformType::Float2),
            ],
            textures: vec!["_texture".to_owned()],
            ..Default::default()
        },
    )
    .unwrap();

    loop {
        clear_background(WHITE);

        material.set_uniform("_resolution", (screen_width(), screen_height()));
        material.set_texture("_texture", render_target.texture);

        gl_use_material(material);
        draw_rectangle(0., 0., screen_width(), screen_height(), WHITE);
        gl_use_default_material();

        next_frame().await
    }
}

const FRAGMENT_SHADER: &str = r#"#version 100
precision lowp float;
varying vec2 uv;
varying vec2 uv_screen;
varying vec2 center;
varying vec2 pixel_size;
uniform sampler2D _texture;
void main() {
    gl_FragColor = texture2D(_texture, uv);
}
"#;

const VERTEX_SHADER: &str = "#version 100
attribute vec3 position;
attribute vec2 texcoord;
varying lowp vec2 center;
varying lowp vec2 uv;
varying lowp vec2 uv_screen;
uniform mat4 Model;
uniform mat4 Projection;
uniform vec2 _resolution;
void main() {
    float coef = max(_resolution.x, _resolution.y);
    vec4 res = Projection * Model * vec4(position, 1);
    uv_screen = res.xy / 2.0 + vec2(0.5, 0.5);
    uv = texcoord * _resolution / coef;
    gl_Position = res;
}
";
