use triangulate::Triangulate;
use macroquad::prelude::*;

#[derive(Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point {
    x: f32,
    y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self { Point {x, y} }
}

impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl triangulate::Vertex for Point {
    type Coordinate = f32;

    #[inline(always)]
    fn x(&self) -> Self::Coordinate { self.x }

    #[inline(always)]
    fn y(&self) -> Self::Coordinate { self.y }
}

impl Into<Point> for (f32, f32) {
    fn into(self) -> Point {
        Point::new(self.0, self.1)
    }
}

#[macroquad::main("Fractal")]
async fn main() {
    let poly = vec![(100.0, 100.0).into(), (300.0, 100.0).into(), (300.0, 300.0).into(), (200.0, 400.0).into(), (100.0, 300.0).into()];

    let polygons: Vec<Vec<Point>> = vec![poly];
    // `output` is an arbitrary triangulation of polygons in a format determined by the type parameter (in this case, a `Vec` of triangle fans represented by a `Vec` of the `MyVert` vertices).
    let output = polygons.triangulate::<triangulate::builders::VecVecFanBuilder<_>>(&mut Vec::new()).unwrap().to_vec();

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

        for i in output {
            draw_triangle((i[0].x, i[0].y).into(), (i[1].x, i[1].y).into(), (i[2].x, i[2].y).into(), GREEN)
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
