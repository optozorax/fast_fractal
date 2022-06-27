use egui_macroquad::egui;
use macroquad::prelude::*;
use triangulate::Triangulate;

#[derive(Debug, Clone, PartialOrd, PartialEq)]
struct MyVec<'a>(&'a DVec2);

impl<'a> triangulate::Vertex for MyVec<'a> {
    type Coordinate = f64;

    #[inline(always)]
    fn x(&self) -> Self::Coordinate {
        self.0.x
    }

    #[inline(always)]
    fn y(&self) -> Self::Coordinate {
        self.0.y
    }
}

impl<'a> From<&'a DVec2> for MyVec<'a> {
    fn from(vec: &'a DVec2) -> Self {
        MyVec(vec)
    }
}

fn to_mat4(mat3: DMat3) -> Mat4 {
    let cols = mat3.to_cols_array();
    Mat4::from_cols(
        Vec4::new(cols[0] as f32, cols[1] as f32, cols[2] as f32, 0.),
        Vec4::new(cols[3] as f32, cols[4] as f32, cols[5] as f32, 0.),
        Vec4::new(cols[6] as f32, cols[7] as f32, cols[8] as f32, 0.),
        Vec4::new(0., 0., 0., 0.),
    )
}

// Assumes that poly points defined in clockwise
fn edge_mat(poly: &[DVec2], base_index: usize, edge_index: usize) -> DMat3 {
    let len_base = (poly[base_index] - poly[(base_index + 1) % poly.len()]).length();
    let b = poly[edge_index];
    let a = poly[(edge_index + 1) % poly.len()];
    let len_ab = (b - a).length();
    let coef = len_ab / len_base;

    let x = DVec3::new(b.x - a.x, b.y - a.y, 0.).normalize() * coef;
    let y = DVec3::new(-(b.y - a.y), b.x - a.x, 0.).normalize() * coef;
    let pos = DVec3::new(a.x, a.y, 1.);

    DMat3::from_cols(x, y, pos).inverse()
}

fn triangulate(poly: &[DVec2]) -> Vec<(DVec2, DVec2, DVec2)> {
    let polygons: Vec<Vec<MyVec>> = vec![poly.iter().map(MyVec).collect()];
    polygons
        .triangulate::<triangulate::builders::VecVecFanBuilder<_>>(&mut Vec::new())
        .unwrap()
        .iter()
        .flat_map(|x| {
            if x.len() == 4 {
                vec![
                    (x[0].0.to_owned(), x[1].0.to_owned(), x[2].0.to_owned()),
                    (x[0].0.to_owned(), x[2].0.to_owned(), x[3].0.to_owned()),
                ]
                .into_iter()
            } else {
                vec![(x[0].0.to_owned(), x[1].0.to_owned(), x[2].0.to_owned())].into_iter()
            }
        })
        .collect()
}

fn draw_polygon(
    poly: &[DVec2],
    mat_poly: DMat3,
    render_target: &mut macroquad::texture::RenderTarget,
    render_target_size: f32,
) {
    set_camera(&Camera2D {
        zoom: vec2(2. / render_target_size, 2. / render_target_size),
        target: vec2(render_target_size / 2., render_target_size / 2.),
        render_target: Some(*render_target),
        ..Default::default()
    });

    clear_background(BLACK);

    for i in triangulate(poly) {
        draw_triangle(
            (mat_poly.inverse() * DVec3::new(i.1.x, i.1.y, 1.)).xy().as_f32(),
            (mat_poly.inverse() * DVec3::new(i.0.x, i.0.y, 1.)).xy().as_f32(),
            (mat_poly.inverse() * DVec3::new(i.2.x, i.2.y, 1.)).xy().as_f32(),
            Color::from_rgba(255, 0, 0, 255),
        );
    }

    set_default_camera();
}

#[allow(clippy::too_many_arguments)]
fn draw_recursive(
    polygon_texture: Texture2D,
    polygon_mat: DMat3,
    prev_texture: Texture2D,
    prev_mat: DMat3,
    draw_target: RenderTarget,
    draw_mat: DMat3,
    draw_target_size: f32,
    mat1: DMat3,
    mat2: DMat3,
    material: Material,
) {
    set_camera(&Camera2D {
        zoom: vec2(2. / draw_target_size, 2. / draw_target_size),
        target: vec2(draw_target_size, draw_target_size) / 2.,
        render_target: Some(draw_target),
        ..Default::default()
    });

    clear_background(BLACK);

    material.set_uniform("_resolution", (draw_target_size, draw_target_size));
    material.set_uniform("_texture_size", draw_target_size);
    material.set_texture("_screen", draw_target.texture);

    material.set_texture("_texture", polygon_texture);
    material.set_uniform("_matrix", to_mat4(polygon_mat.inverse() * draw_mat));
    gl_use_material(material);
    draw_rectangle(0., 0., draw_target_size, draw_target_size, WHITE);
    gl_use_default_material();

    set_default_camera();

    set_camera(&Camera2D {
        zoom: vec2(2. / draw_target_size, 2. / draw_target_size),
        target: vec2(draw_target_size, draw_target_size) / 2.,
        render_target: Some(draw_target),
        ..Default::default()
    });
    material.set_texture("_texture", prev_texture);

    material.set_uniform("_matrix", to_mat4(prev_mat.inverse() * mat1 * draw_mat));
    gl_use_material(material);
    draw_rectangle(0., 0., draw_target_size, draw_target_size, WHITE);
    gl_use_default_material();

    set_default_camera();

    set_camera(&Camera2D {
        zoom: vec2(2. / draw_target_size, 2. / draw_target_size),
        target: vec2(draw_target_size, draw_target_size) / 2.,
        render_target: Some(draw_target),
        ..Default::default()
    });
    material.set_texture("_texture", prev_texture);

    material.set_uniform("_matrix", to_mat4(prev_mat.inverse() * mat2 * draw_mat));
    gl_use_material(material);
    draw_rectangle(0., 0., draw_target_size, draw_target_size, WHITE);
    gl_use_default_material();

    set_default_camera();
}

pub fn toggle_ui(ui: &mut egui::Ui, pos: &mut DVec2, coef_x: f64, coef_y: f64) -> bool {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(1.0, 1.0);

    let rect = egui::Rect::from_min_size(
        egui::pos2((pos.x * coef_x) as f32, (pos.y * coef_y) as f32) - desired_size / 2.,
        desired_size,
    );
    let mut response = ui.allocate_rect(rect, egui::Sense::drag());

    let mut changed = false;
    if response.dragged() {
        ui.output().cursor_icon = egui::CursorIcon::Move;
        let delta = response.drag_delta();
        pos.x += delta.x as f64 / coef_x;
        pos.y += delta.y as f64 / coef_y;
        response.mark_changed();
        changed = true;
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui
            .style()
            .interact_selectable(&response, response.dragged());
        let radius = 0.5 * rect.height();
        let rect = rect.expand(visuals.expansion);
        let center = egui::pos2(rect.center().x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }

    changed
}

#[macroquad::main("Fractal")]
async fn main() {
    let mut poly: Vec<DVec2> = vec![
        (0.0, 0.0).into(),
        (400.0, 0.0).into(),
        (400.5, 200.5).into(),
        (100.5, 300.5).into(),
        (0.0, 200.0).into(),
    ];

    let scale = 4.;

    let mat_poly = (DMat3::from_scale(DVec2::new(scale, scale)) * DMat3::from_translation(DVec2::new(50., 25.))).inverse(); // translation of original polygon
    let mat_fractal = (DMat3::from_scale(DVec2::new(scale, scale)) * DMat3::from_translation(DVec2::new(200., 100.))).inverse(); // translation of full fractal

    let size = (1000. * scale) as u32;
    let sizef = size as f32;

    let material = load_material(
        VERTEX_SHADER,
        FRAGMENT_SHADER,
        MaterialParams {
            uniforms: vec![
                ("_texture_size".to_owned(), UniformType::Float1),
                ("_resolution".to_owned(), UniformType::Float2),
                ("_matrix".to_owned(), UniformType::Mat4),
            ],
            textures: vec!["_texture".to_owned(), "_screen".to_owned()],
            ..Default::default()
        },
    )
    .unwrap();

    let mut render_target = render_target(size, size);
    let mut screen1 = macroquad::prelude::render_target(size, size);
    let mut screen2 = macroquad::prelude::render_target(size, size);
    let mut mat1 = edge_mat(&poly, 0, 2);
    let mut mat2 = edge_mat(&poly, 0, 3);

    let mut changed = true;

    // let mut now = std::time::Instant::now();

    loop {
        // println!("{:5.1}FPS", 1000. / (now.elapsed().as_millis()) as f32);
        // now = std::time::Instant::now();

        egui_macroquad::ui(|egui_ctx| {
            let available_rect = egui_ctx.available_rect();
            let layer_id = egui::LayerId::background();
            let id = egui::Id::new("central_panel");

            let clip_rect = egui_ctx.input().screen_rect();
            let mut panel_ui =
                egui::Ui::new(egui_ctx.clone(), layer_id, id, available_rect, clip_rect);

            for i in &mut poly {
                let mut pos = (mat_fractal.inverse() * DVec3::new(i.x, i.y, 1.)).xy();
                let this_changed = toggle_ui(
                    &mut panel_ui,
                    &mut pos,
                    (screen_width() / sizef) as f64,
                    (screen_height() / sizef) as f64,
                );
                changed |= this_changed;
                if this_changed {
                    *i = (mat_fractal * DVec3::new(pos.x, pos.y, 1.)).xy();
                }
            }
        });


        if changed {
            mat1 = edge_mat(&poly, 0, 2);
            mat2 = edge_mat(&poly, 0, 3);

            draw_polygon(&poly, mat_poly, &mut render_target, sizef);

            changed = false;
        }

        draw_recursive(
            render_target.texture,
            mat_poly,
            screen1.texture,
            mat_fractal,
            screen2,
            mat_fractal,
            sizef,
            mat1,
            mat2,
            material,
        );
        std::mem::swap(&mut screen1, &mut screen2);

        draw_texture_ex(
            screen2.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        egui_macroquad::draw();

        next_frame().await;
    }
}

const FRAGMENT_SHADER: &str = r#"#version 100
precision lowp float;
varying vec2 uv;
varying vec2 uv_screen;
varying vec2 center;
varying float pixel_size;
varying vec4 color;
uniform sampler2D _texture;
uniform float _texture_size;
uniform sampler2D _screen;
void main() {
    vec3 c = texture2D(_screen, uv_screen).xyz;
    if (uv.x > 0. && uv.y > 0. && uv.x < 1.0 && uv.y < 1.0) {
        int aa_size = 2;
        for (int i = 0; i < aa_size; ++i) {
            for (int j = 0; j < aa_size; ++j) {
                vec2 uv_offset = vec2(float(i), float(j)) / float(aa_size) * pixel_size;
                vec3 getted = texture2D(_texture, uv + uv_offset).xyz;
                c += getted / float(aa_size * aa_size);
            }
        }
        if (c.x > 1.) {
            c.y += (c.x - 1.) / 4.;
            c.x = 1.;
        }
        if (c.y > 1.) {
            c.z += (c.y - 1.) / 4.;
            c.y = 1.;
        }
        if (c.z > 1.) {
            c.z = 1.;
        }
    }
    gl_FragColor = vec4(c, 1.);
}
"#;

const VERTEX_SHADER: &str = "#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
varying lowp vec2 center;
varying lowp vec2 uv;
varying lowp vec2 uv_screen;
varying lowp vec4 color;
varying float pixel_size;
uniform float _texture_size;
uniform mat4 Model;
uniform mat4 Projection;
uniform mat4 _matrix;
uniform vec2 _resolution;
void main() {
    float coef = max(_resolution.x, _resolution.y);
    pixel_size = 1.0 / coef;
    vec4 res = Projection * Model * vec4(position, 1);
    uv_screen = res.xy / 2.0 + vec2(0.5, 0.5);
    uv = (_matrix * vec4((texcoord * _texture_size * _resolution / coef).xy, 1.0, 0.)).xy / _texture_size;
    color = color0 / 255.0;
    gl_Position = res;
}
";
