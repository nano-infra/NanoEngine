//! Utility handlers: health check, Redis address resolution.

use axum::{extract::State, response::IntoResponse, Json};

use crate::error::AppError;
use crate::models::*;
use crate::net;
use crate::redis_repo::RedisRepo;

pub async fn root() -> &'static str {
    "NanoCtrl Server Running"
}

pub async fn get_redis_address(
    State(repo): State<RedisRepo>,
    Json(_body): Json<GetRedisAddressBody>,
) -> Result<impl IntoResponse, AppError> {
    let redis_address = net::resolve_public_redis_url(repo.redis_url());
    tracing::debug!(
        "Returning Redis URL: {} (original: {})",
        redis_address,
        repo.redis_url()
    );
    Ok(Json(GetRedisAddressResponse {
        status: "ok".to_string(),
        redis_address,
    }))
}
