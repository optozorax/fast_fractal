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

fn get_line(poly: &[DVec2], index: usize) -> (DVec2, DVec2) {
    (poly[index % poly.len()], poly[(index + 1) % poly.len()])
}

fn inverse_line((a, b): (DVec2, DVec2)) -> (DVec2, DVec2) {
    (b, a)
}

fn line_len((a, b): (DVec2, DVec2)) -> f64 {
    (b - a).length()
}

fn is_draw_here(poly: &[DVec2], base_index: usize, edge_index: usize) -> bool {
    line_len(get_line(poly, edge_index)) <= line_len(get_line(poly, base_index))
}

// Return matrix that corresponds to coordinate system, with OX equals to these two points, and OY perpendicular to that, and with O equals to first point
fn two_point_mat((a, b): (DVec2, DVec2)) -> DMat3 {
    let x = DVec3::new(b.x - a.x, b.y - a.y, 0.);
    let y = DVec3::new(-(b.y - a.y), b.x - a.x, 0.);
    let pos = DVec3::new(a.x, a.y, 1.);

    DMat3::from_cols(x, y, pos)
}

// Assumes that poly points defined in clockwise
fn edge_mat(poly: &[DVec2], base_index: usize, edge_index: usize) -> DMat3 {
    edge_mat_old(poly, get_line(poly, base_index), edge_index)
}

// This method called when base points is changed, to preserve them, to seamessly transform between different bases
fn edge_mat_old(poly: &[DVec2], base: (DVec2, DVec2), edge_index: usize) -> DMat3 {
    two_point_mat(base) * two_point_mat(inverse_line(get_line(poly, edge_index))).inverse()
}

fn triangulate(poly: &[DVec2]) -> Option<Vec<(DVec2, DVec2, DVec2)>> {
    let polygons: Vec<Vec<MyVec>> = vec![poly.iter().map(MyVec).collect()];
    polygons
        .triangulate::<triangulate::builders::VecVecFanBuilder<_>>(&mut Vec::new())
        .ok()?
        .iter()
        .map(|x| {
            if x.len() >= 4 {
                None
            } else {
                Some((x[0].0.to_owned(), x[1].0.to_owned(), x[2].0.to_owned()))
            }
        })
        .collect()
}

fn draw_polygon(
    poly: &[DVec2],
    mat_poly: DMat3,
    render_target: &mut macroquad::texture::RenderTarget,
    render_target_size: f32,
) -> bool {
    set_camera(&Camera2D {
        zoom: vec2(2. / render_target_size, 2. / render_target_size),
        target: vec2(render_target_size / 2., render_target_size / 2.),
        render_target: Some(*render_target),
        ..Default::default()
    });

    clear_background(BLACK);

    let color = Color::from_rgba(1, 0, 255, 255);

    if let Some(triangulation) = triangulate(poly) {
        for i in triangulation {
            draw_triangle(
                (mat_poly.inverse().transform_point2(i.1)).xy().as_f32(),
                (mat_poly.inverse() * DVec3::new(i.0.x, i.0.y, 1.))
                    .xy()
                    .as_f32(),
                (mat_poly.inverse() * DVec3::new(i.2.x, i.2.y, 1.))
                    .xy()
                    .as_f32(),
                color,
            );
        }

        set_default_camera();

        true
    } else {
        for i in 0..poly.len() {
            let (a, b) = get_line(poly, i);
            let a = mat_poly.inverse().transform_point2(a);
            let b = mat_poly.inverse().transform_point2(b);
            draw_line(
                a.x as f32,
                a.y as f32,
                b.x as f32,
                b.y as f32,
                render_target_size / 200.,
                color,
            );
        }

        set_default_camera();

        false
    }
}

fn draw_on_target<F: FnMut()>(on: RenderTarget, size: f32, mut f: F) {
    set_camera(&Camera2D {
        zoom: vec2(2. / size, 2. / size),
        target: vec2(size, size) / 2.,
        render_target: Some(on),
        ..Default::default()
    });

    f();

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
    mats: &[DMat3],
    material: Material,
    is_draw_mats: &[bool],
) {
    draw_on_target(draw_target, draw_target_size, || {
        clear_background(BLACK);

        material.set_uniform("_resolution", (draw_target_size, draw_target_size));
        material.set_uniform("_texture_size", draw_target_size);
        material.set_texture("_screen", draw_target.texture);

        material.set_texture("_texture", polygon_texture);
        material.set_uniform("_matrix", to_mat4(polygon_mat.inverse() * draw_mat));
        gl_use_material(material);
        draw_rectangle(0., 0., draw_target_size, draw_target_size, WHITE);
        gl_use_default_material();
    });

    for (is_draw, mat) in is_draw_mats.iter().zip(mats.iter()) {
        if *is_draw {
            draw_on_target(draw_target, draw_target_size, || {
                material.set_texture("_texture", prev_texture);

                material.set_uniform("_matrix", to_mat4(prev_mat.inverse() * *mat * draw_mat));
                gl_use_material(material);
                draw_rectangle(0., 0., draw_target_size, draw_target_size, WHITE);
                gl_use_default_material();

                set_default_camera();
            });
        }
    }
}

pub fn point_ui(ui: &mut egui::Ui, pos: &mut DVec2, coef_x: f64, coef_y: f64) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(1.0, 1.0);

    let rect = egui::Rect::from_min_size(
        egui::pos2((pos.x * coef_x) as f32, (pos.y * coef_y) as f32) - desired_size / 2.,
        desired_size,
    );
    let mut response = ui.allocate_rect(rect, egui::Sense::drag());

    if response.dragged() {
        ui.output().cursor_icon = egui::CursorIcon::Move;
        let delta = response.drag_delta();
        pos.x += delta.x as f64 / coef_x;
        pos.y += delta.y as f64 / coef_y;
        response.mark_changed();
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

    response
}

pub fn move_ui(ui: &mut egui::Ui, pos: &mut DVec2, coef_x: f64, coef_y: f64) -> bool {
    let rect = egui::Rect::from_min_size(
        egui::pos2(0., 0.),
        egui::vec2(screen_width(), screen_height()),
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

    changed
}

#[derive(Debug, Clone)]
struct BoundingBox {
    start: DVec2,
    min: DVec2,
    max: DVec2,
}

impl BoundingBox {
    fn new(point: DVec2) -> Self {
        Self {
            start: point,
            min: Default::default(),
            max: Default::default(),
        }
    }

    fn update(&mut self, mut point: DVec2) {
        point -= self.start;

        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);

        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
    }

    // Returns matrix that transforms coordinates inside bounding box into [0; 1]x[0; 1] (preserves aspect ratio)
    fn transform_mat(&self, sizef: f64, padding_percent: f64) -> DMat3 {
        let min = self.start + self.min;
        let max = self.start + self.max;
        let size = max - min;

        let min = min - padding_percent * size;
        let max = max + padding_percent * size;
        let size = max - min;

        let scale = size.x.max(size.y);
        DMat3::from_translation(min) * DMat3::from_scale(DVec2::new(scale / sizef, scale / sizef))
    }

    fn unioni(&mut self, other: &BoundingBox) {
        self.update(other.start + other.min);
        self.update(other.start + other.max);
    }

    fn mul(&mut self, mat: DMat3) {
        self.start = mat.transform_point2(self.start);
        self.min = mat.transform_vector2(self.min);
        self.max = mat.transform_vector2(self.max);
    }

    fn draw(&self, mat: DMat3, color: Color) {
        let min = self.start + self.min;
        let max = self.start + self.max;

        let a = mat.transform_point2(min).as_f32();
        let b = mat.transform_point2(DVec2::new(min.x, max.y)).as_f32();
        let c = mat.transform_point2(max).as_f32();
        let d = mat.transform_point2(DVec2::new(max.x, min.y)).as_f32();

        let thicc = 2.0;
        draw_line(a.x, a.y, b.x, b.y, thicc, color);
        draw_line(b.x, b.y, c.x, c.y, thicc, color);
        draw_line(c.x, c.y, d.x, d.y, thicc, color);
        draw_line(d.x, d.y, a.x, a.y, thicc, color);
        draw_line(a.x, a.y, c.x, c.y, thicc / 2.0, color);
    }
}

const BB_SIZE: usize = 360 * 5;

fn get_rot_mat(index: i64) -> DMat3 {
    DMat3::from_rotation_z((index as f64) / BB_SIZE as f64 * std::f64::consts::TAU)
}

#[derive(Clone, Debug)]
struct BoundingBox360(Vec<BoundingBox>);

impl BoundingBox360 {
    pub fn new(poly: &[DVec2]) -> Self {
        Self(
            (0..BB_SIZE)
                .map(|i| -(i as f64) / BB_SIZE as f64 * std::f64::consts::TAU)
                .map(|angle| {
                    let matrix = DMat2::from_angle(angle);
                    let mut bb = BoundingBox::new(matrix * *poly.first().unwrap());
                    for i in poly {
                        bb.update(matrix * *i);
                    }
                    bb.start = *poly.first().unwrap();
                    bb
                })
                .collect(),
        )
    }

    pub fn mul(&mut self, mat: DMat3) {
        let axis = DVec2::new(1., 0.);
        let mat_axis = mat.transform_vector2(axis);
        let offset = ((std::f64::consts::PI + axis.angle_between(mat_axis)) * BB_SIZE as f64
            / std::f64::consts::TAU) as usize;
        let new_vec = (0..BB_SIZE)
            .map(|i| {
                let index = (i + offset) % BB_SIZE;
                let mut res = self.0[index].clone();
                res.mul(get_rot_mat(-(i as i64)) * mat.inverse() * get_rot_mat(index as i64));
                res
            })
            .collect::<Vec<_>>();
        self.0 = new_vec;
    }

    pub fn unioni(&mut self, other: &Self) {
        for (current, other) in self.0.iter_mut().zip(other.0.iter()) {
            current.unioni(other);
        }
    }

    pub fn draw(&self, mat: DMat3, color: Color) {
        self.0.first().unwrap().draw(mat, color);
    }

    pub fn transform_mat(&self, sizef: f64, padding_percent: f64) -> DMat3 {
        self.0[0].transform_mat(sizef, padding_percent)
    }
}

#[macroquad::main("Fractal")]
async fn main() {
    let mut poly: Vec<DVec2> = vec![
        (0.0, 0.0).into(),
        (200.0, 0.0).into(),
        (200.0, 150.0).into(),
        (100.0, 210.0).into(),
        (0.0, 150.0).into(),
    ];

    let mut draw_on: Vec<usize> = vec![2, 3];
    let mut skip_points: Vec<bool> = Vec::new();

    let mut texture_size: u32 = 2;

    let mut size = 1000 * texture_size;
    let mut sizef = size as f32;

    let mut bb = BoundingBox360::new(&poly);
    let mut poly_bb = bb.clone();

    let padding_percent = 0.1;

    let mut mat_poly = poly_bb.transform_mat(sizef.into(), padding_percent);

    let cam_scale = 0.25;
    let cam_offset = DVec2::new(200., 100.) * texture_size as f64;
    let mut mat_fractal = DMat3::from_scale(DVec2::new(cam_scale, cam_scale))
        * DMat3::from_translation(cam_offset).inverse(); // translation of full fractal
    let mut mat_fractal_next = mat_fractal;

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

    let material_final = load_material(
        VERTEX_SHADER_FINAL,
        FRAGMENT_SHADER_FINAL,
        MaterialParams {
            uniforms: vec![
                ("_texture_size".to_owned(), UniformType::Float1),
                ("_resolution".to_owned(), UniformType::Float2),
                ("_matrix".to_owned(), UniformType::Mat4),
            ],
            textures: vec!["_texture".to_owned()],
            ..Default::default()
        },
    )
    .unwrap();

    let mut render_target = macroquad::prelude::render_target(size, size);
    let mut screen1 = macroquad::prelude::render_target(size, size);
    let mut screen2 = macroquad::prelude::render_target(size, size);

    render_target.texture.set_filter(FilterMode::Nearest);
    screen1.texture.set_filter(FilterMode::Nearest);
    screen2.texture.set_filter(FilterMode::Nearest);

    let mut mats: Vec<DMat3> = draw_on.iter().map(|i| edge_mat(&poly, 0, *i)).collect();

    let mut changed = true;
    let mut changed2 = false;

    let mut triangulation_succeded = true;
    let mut sides_too_big = false;

    let mut mat_bb_prev = bb.transform_mat(sizef.into(), padding_percent);

    let mut old_points;

    let mut screen_size;

    let mut draw_bounding_box = false;

    loop {
        mat_fractal = mat_fractal_next;
        old_points = get_line(&poly, 0);
        screen_size = screen_width().max(screen_height());

        egui_macroquad::ui(|egui_ctx| {
            let available_rect = egui_ctx.available_rect();
            let layer_id = egui::LayerId::background();
            let id = egui::Id::new("central_panel");

            let clip_rect = egui_ctx.input().screen_rect();
            let mut panel_ui =
                egui::Ui::new(egui_ctx.clone(), layer_id, id, available_rect, clip_rect);

            for (index, i) in poly.iter_mut().enumerate() {
                let mut pos = (mat_fractal.inverse() * DVec3::new(i.x, i.y, 1.)).xy();
                let response = point_ui(
                    &mut panel_ui,
                    &mut pos,
                    (screen_size / sizef) as f64,
                    (screen_size / sizef) as f64,
                );
                if response.dragged() {
                    egui::show_tooltip(egui_ctx, egui::Id::new("my_tooltip"), |ui| {
                        ui.label(format!("Point #{}", index));
                        if !triangulation_succeded {
                            ui.label("ERROR: Triangulation failed");
                        }
                        if sides_too_big {
                            ui.label("ERROR: Sides too big");
                        }
                    });
                }
                changed |= response.changed();
                if response.changed() {
                    *i = (mat_fractal * DVec3::new(pos.x, pos.y, 1.)).xy();
                }
            }

            let mut offset = DVec2::default();
            if move_ui(
                &mut panel_ui,
                &mut offset,
                (screen_size / sizef) as f64,
                (screen_size / sizef) as f64,
            ) {
                mat_fractal_next = mat_fractal_next * DMat3::from_translation(-offset);
            }

            {
                let wheel = egui_ctx.input().scroll_delta;
                if wheel.y != 0. {
                    let scale = 1.02_f64.powf(wheel.y as f64);
                    let pos = egui_ctx.input().pointer.interact_pos();
                    if let Some(pos) = pos {
                        let pos2 = DVec2::new(
                            pos.x as f64 / (screen_size / sizef) as f64,
                            pos.y as f64 / (screen_size / sizef) as f64,
                        );
                        mat_fractal_next = mat_fractal_next
                            * DMat3::from_translation(pos2)
                            * DMat3::from_scale(DVec2::new(scale, scale))
                            * DMat3::from_translation(-pos2);
                    }
                }
            }

            egui::Window::new("settings")
                .default_width(200.0)
                .show(egui_ctx, |ui| {
                    ui.label("Polygon:");
                    ui.horizontal(|ui| {
                        if ui.button("Add point").clicked() {
                            let (a, b) = get_line(&poly, poly.len() - 1);
                            poly.push((b + a) / 2.);
                            changed = true;
                        }
                        if ui
                            .add_enabled(poly.len() > 3, egui::Button::new("Remove point"))
                            .clicked()
                        {
                            poly.pop();
                            changed = true;
                        }
                    });
                    ui.separator();
                    egui::Grid::new("my_grid")
                        .num_columns(3)
                        .spacing([40.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for (index, i) in poly.iter_mut().enumerate() {
                                ui.label(format!("{index}"));
                                changed |=
                                    ui.add(egui::DragValue::new(&mut i.x).speed(0.1)).changed();
                                changed |=
                                    ui.add(egui::DragValue::new(&mut i.y).speed(0.1)).changed();
                                ui.end_row();
                            }
                        });
                    ui.separator();
                    {
                        let mut bool_draw_on = vec![false; poly.len()];
                        for i in &draw_on {
                            if bool_draw_on.len() > *i {
                                bool_draw_on[*i] = true;
                            }
                        }
                        bool_draw_on[0] = false;
                        ui.horizontal(|ui| {
                            ui.label("Draw on sides: ");
                            for (index, b) in bool_draw_on.iter_mut().enumerate().skip(1) {
                                let old = *b;
                                let previous = ui.visuals().clone();
                                let show_error_color = *b && !is_draw_here(&poly, 0, index);
                                if show_error_color {
                                    ui.visuals_mut().selection.stroke.color = egui::Color32::RED;
                                    ui.visuals_mut().widgets.inactive.bg_stroke.color =
                                        egui::Color32::from_rgb_additive(128, 0, 0);
                                    ui.visuals_mut().widgets.inactive.bg_stroke.width = 1.0;
                                    ui.visuals_mut().widgets.hovered.bg_stroke.color =
                                        egui::Color32::from_rgb_additive(255, 128, 128);
                                }
                                ui.toggle_value(b, format!("{index}"));
                                if show_error_color {
                                    *ui.visuals_mut() = previous;
                                }
                                changed |= old != *b;
                            }
                        });
                        draw_on = bool_draw_on
                            .into_iter()
                            .enumerate()
                            .filter_map(|(index, enabled)| if enabled { Some(index) } else { None })
                            .collect();
                    }
                    ui.separator();
                    {
                        skip_points.resize(poly.len(), false);
                        ui.horizontal(|ui| {
                            ui.label("Skip points: ");
                            for (index, b) in skip_points.iter_mut().enumerate() {
                                let old = *b;
                                ui.toggle_value(b, format!("{index}"));
                                changed |= old != *b;
                            }
                        });
                    }
                    ui.separator();
                    ui.checkbox(&mut draw_bounding_box, "Draw bounding box");
                    ui.separator();
                    if ui.button("Change cam to see full fractal").clicked() {
                        mat_fractal_next = bb.transform_mat(sizef.into(), 0.03);
                    }
                    ui.separator();
                    ui.label("Texture size: ");
                    for j in 0..3 {
                        ui.horizontal(|ui| {
                            for i in (j * 5 + 1)..((j + 1) * 5 + 1) {
                                if ui
                                    .selectable_label(texture_size == i, format!("{:2}k²", i))
                                    .clicked()
                                {
                                    let coef = texture_size as f64 / i as f64;
                                    mat_fractal_next =
                                        mat_fractal * DMat3::from_scale(DVec2::new(coef, coef));
                                    texture_size = i;
                                    size = texture_size * 1000;
                                    sizef = size as f32;
                                    render_target.delete();
                                    screen1.delete();
                                    screen2.delete();
                                    render_target = macroquad::prelude::render_target(size, size);
                                    screen1 = macroquad::prelude::render_target(size, size);
                                    screen2 = macroquad::prelude::render_target(size, size);
                                    render_target.texture.set_filter(FilterMode::Nearest);
                                    screen1.texture.set_filter(FilterMode::Nearest);
                                    screen2.texture.set_filter(FilterMode::Nearest);
                                    draw_on_target(render_target, sizef, || {
                                        clear_background(BLACK)
                                    });
                                    draw_on_target(screen1, sizef, || clear_background(BLACK));
                                    draw_on_target(screen2, sizef, || clear_background(BLACK));
                                    changed = true;
                                }
                            }
                        });
                    }
                });
        });

        if changed {
            changed2 = false;
        }

        if changed || changed2 {
            if changed2 {
                mats = draw_on.iter().map(|i| edge_mat(&poly, 0, *i)).collect();
                changed2 = false;
            } else {
                mats = draw_on
                    .iter()
                    .map(|i| edge_mat_old(&poly, old_points, *i))
                    .collect();
            }

            mat_poly = poly_bb.transform_mat(sizef.into(), padding_percent);

            let poly_skipped = poly
                .iter()
                .zip(skip_points.iter())
                .filter(|(_, skip)| !**skip)
                .map(|(point, _)| *point)
                .collect::<Vec<_>>();
            triangulation_succeded =
                draw_polygon(&poly_skipped, mat_poly, &mut render_target, sizef);
            sides_too_big = !draw_on
                .iter()
                .map(|i| is_draw_here(&poly, 0, *i))
                .all(|i| i);

            poly_bb = BoundingBox360::new(&poly);

            if changed {
                changed = false;
                changed2 = true;
            }
        }

        let mut bb_arr_full = poly_bb.clone();
        for (draw_on_pos, mat) in draw_on.iter().zip(mats.iter()) {
            if is_draw_here(&poly, 0, *draw_on_pos) {
                let mut bb_arr_mat = bb.clone();
                bb_arr_mat.mul(*mat);
                bb_arr_full.unioni(&bb_arr_mat);
            }
        }
        std::mem::swap(&mut bb, &mut bb_arr_full);

        let mat_bb = bb.transform_mat(sizef.into(), padding_percent);

        draw_recursive(
            render_target.texture,
            mat_poly,
            screen1.texture,
            mat_bb_prev,
            screen2,
            mat_bb,
            sizef,
            &mats,
            material,
            &draw_on
                .iter()
                .map(|i| is_draw_here(&poly, 0, *i))
                .collect::<Vec<_>>(),
        );
        std::mem::swap(&mut screen1, &mut screen2);

        material_final.set_uniform("_texture_size", sizef);
        material_final.set_uniform("_resolution", (sizef, sizef));
        material_final.set_uniform("_matrix", to_mat4(mat_bb_prev.inverse() * mat_fractal));
        material_final.set_texture("_texture", screen2.texture);
        gl_use_material(material_final);
        draw_rectangle(0., 0., screen_size, screen_size, WHITE);
        gl_use_default_material();

        mat_bb_prev = mat_bb;

        if draw_bounding_box {
            let bb_draw_mat = DMat3::from_scale(DVec2::new(
                (screen_size / sizef).into(),
                (screen_size / sizef).into(),
            )) * mat_fractal.inverse();
            bb.draw(bb_draw_mat, BLUE);
        }

        egui_macroquad::draw();

        next_frame().await;

        // std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

const FRAGMENT_SHADER: &str = r#"#version 100
precision highp float;

varying vec2 uv;
varying vec2 uv_screen;
varying vec4 color;
varying float pixel_size;

uniform float _texture_size;
uniform sampler2D _texture;
uniform sampler2D _screen;

void main() {
    vec3 c = texture2D(_screen, uv_screen).xyz;
    if (uv.x > 0. && uv.y > 0. && uv.x < 1.0 && uv.y < 1.0) {
        vec3 c1 = texture2D(_texture, uv).xyz;
        if (c1.z > 0.5) { 
            c = vec3(1./256. , 0., 0.1);
        } else if (c1.x > 0. || c1.y > 0.) { 
            c1.x += 1./256.;
            if (c1.x >= 1.) {
                c1.x = 0.;
                c1.y += 1./256.;
            }
            if (c1.y >= 1.) {
                c1.y = 1.;
            }
            if (c.x > 0. || c.y > 0.) {
                // if (c1.y > c.y || (c1.y == c.y && c1.x > c.x)) {
                if (c1.y < c.y || (c1.y == c.y && c1.x < c.x)) {
                    c = c1;
                }
            } else {
                c = c1;
            }
        }
    }
    gl_FragColor = vec4(c, 1.);
}
"#;

const VERTEX_SHADER: &str = "#version 100
precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying vec2 uv;
varying vec2 uv_screen;
varying vec4 color;
varying float pixel_size;

uniform float _texture_size;
uniform mat4 Model;
uniform mat4 Projection;
uniform mat4 _matrix;
uniform vec2 _resolution;

void main() {
    vec4 res = Projection * Model * vec4(position, 1);
    uv_screen = res.xy / 2.0 + vec2(0.5, 0.5);
    float coef = max(_resolution.x, _resolution.y);
    pixel_size = 1.0 / coef;
    uv = (_matrix * vec4((texcoord * _texture_size * _resolution / coef).xy, 1.0, 0.)).xy / _texture_size;
    color = color0 / 255.0;
    gl_Position = res;
}
";

const FRAGMENT_SHADER_FINAL: &str = r#"#version 100
precision highp float;

varying vec2 uv;

uniform sampler2D _texture;

#define SRGB_TO_LINEAR(c) pow((c), vec3(2.2))
#define LINEAR_TO_SRGB(c) pow((c), vec3(1.0 / 2.2))
#define SRGB(r, g, b) SRGB_TO_LINEAR(vec3(float(r), float(g), float(b)) / 255.0)

const vec3 COLOR0 = SRGB(255, 0, 114);
const vec3 COLOR1 = SRGB(197, 255, 80);
const vec3 COLOR2 = SRGB(0, 128, 192);
const vec3 COLOR3 = SRGB(0, 230, 230);
const vec3 COLOR4 = SRGB(240, 240, 240);

float modI(float a, float b) {
    float m=a-floor((a+0.5)/b)*b;
    return floor(m+0.5);
}

void main() {
    vec3 c = texture2D(_texture, uv).xyz;
    float depth = (c.x + c.y * 256.) * 256.;
    if (c.z != 0.) {
        float step = 6.;
        c = COLOR0;

        float depthPosStep = floor(depth / step);
        float i = floor(depth - depthPosStep * step);
        int j = int(modI(depthPosStep, 5.));
        {
            vec3 a = COLOR0;
            if (j == 0)  a = COLOR0;
            if (j == 1)  a = COLOR1;
            if (j == 2)  a = COLOR2;
            if (j == 3)  a = COLOR3;
            if (j == 4)  a = COLOR4;

            vec3 b = COLOR1;
            if (j == 0)  b = COLOR1;
            if (j == 1)  b = COLOR2;
            if (j == 2)  b = COLOR3;
            if (j == 3)  b = COLOR4;
            if (j == 4)  b = COLOR0;
            c = LINEAR_TO_SRGB(mix(a, b, clamp(float(i) / step, 0., 1.)));
        }
    }
    gl_FragColor = vec4(c, 1.);
}
"#;

const VERTEX_SHADER_FINAL: &str = "#version 100
precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying vec2 uv;

uniform float _texture_size;
uniform mat4 Model;
uniform mat4 Projection;
uniform mat4 _matrix;
uniform vec2 _resolution;

void main() {
    vec4 res = Projection * Model * vec4(position, 1);
    float coef = max(_resolution.x, _resolution.y);
    // pixel_size = 1.0 / coef;
    uv = (_matrix * vec4((texcoord * _texture_size * _resolution / coef).xy, 1.0, 0.)).xy / _texture_size;
    gl_Position = res;
}
";
