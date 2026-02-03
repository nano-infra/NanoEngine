//! Owned Peer type and conversion from/to flatbuffers Peer table.
//! The flatc we use does not generate object API (PeerT with unpack/pack), so we provide it here.

use crate::fbs::{
    EndpointBytes, EndpointBytesArgs, EndpointInfoList, EndpointInfoListArgs, Peer, PeerArgs,
};
use flatbuffers::FlatBufferBuilder;

/// Owned peer info used for P2P init/connect (replaces missing flatbuffers object-api PeerT).
#[derive(Clone, Default)]
pub struct PeerT {
    pub remote_id: Option<String>,
    /// remote_info: [EndpointInfoList], each list has endpoints: [EndpointBytes], each has data: [u8]
    pub remote_info: Option<Vec<Vec<Vec<u8>>>>,
}

/// Build owned PeerT from a flatbuffers Peer table.
pub fn peer_from_table<'a>(peer: Peer<'a>) -> PeerT {
    let remote_id = peer.remote_id().map(|s| s.to_string());
    let remote_info = peer.remote_info().map(|vec| {
        let mut list = Vec::new();
        for i in 0..vec.len() {
            let el = vec.get(i);
            let mut blobs = Vec::new();
            if let Some(ep_vec) = el.endpoints() {
                for j in 0..ep_vec.len() {
                    if let Some(slice) = ep_vec.get(j).data() {
                        blobs.push(slice.to_vec());
                    }
                }
            }
            list.push(blobs);
        }
        list
    });
    PeerT {
        remote_id,
        remote_info,
    }
}

impl PeerT {
    /// Serialize this peer into the builder and return WIPOffset<Peer>.
    pub fn pack<'a>(
        &self,
        builder: &mut FlatBufferBuilder<'a>,
    ) -> flatbuffers::WIPOffset<Peer<'a>> {
        let remote_id_off = self.remote_id.as_ref().map(|s| builder.create_string(s));
        let remote_info_off = self.remote_info.as_ref().and_then(|lists| {
            let list_offsets: Vec<_> = lists
                .iter()
                .map(|endpoints| {
                    let eb_offsets: Vec<_> = endpoints
                        .iter()
                        .map(|data| {
                            let data_off = builder.create_vector(data);
                            EndpointBytes::create(
                                builder,
                                &EndpointBytesArgs {
                                    data: Some(data_off),
                                },
                            )
                        })
                        .collect();
                    let vec_off = builder.create_vector(&eb_offsets);
                    EndpointInfoList::create(
                        builder,
                        &EndpointInfoListArgs {
                            endpoints: Some(vec_off),
                        },
                    )
                })
                .collect();
            if list_offsets.is_empty() {
                None
            } else {
                Some(builder.create_vector(&list_offsets))
            }
        });
        Peer::create(
            builder,
            &PeerArgs {
                remote_id: remote_id_off,
                remote_info: remote_info_off,
            },
        )
    }
}
