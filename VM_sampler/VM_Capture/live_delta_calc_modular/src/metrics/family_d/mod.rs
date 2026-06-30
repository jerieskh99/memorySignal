//! Family D -- internal structure / texture: where the change sits inside the page,
//! and how textured the new content is. Two sub-families.

pub mod change_location;
pub mod texture;

#[derive(Clone, Default, Debug)]
pub struct Internal {
    pub change_location: change_location::ChangeLocation,
    pub texture: texture::Texture,
}

pub fn compute(p: &[u8], q: &[u8]) -> Internal {
    Internal {
        change_location: change_location::compute(p, q),
        texture: texture::compute(q),
    }
}
