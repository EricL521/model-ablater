SHOWN_LAYERS = {
	"ln1.hook_normalized": None, 
	"attn.hook_q": "current.attn.hook_q", 
	"attn.hook_k": "current.attn.hook_k", 
	"attn.hook_v": "current.attn.hook_v", 
	"attn.hook_rot_q": "current.attn.hook_q", 
	"attn.hook_rot_k": "current.attn.hook_k", 
	"attn.hook_pattern": None, 
	"attn.hook_z": "current.attn.hook_v", 
	"hook_attn_out": None, 
	"hook_resid_mid": None, 
	"ln2.hook_normalized": None, 
	"mlp.hook_pre": "current.mlp.hook_pre", # NOTE: this is the gate activation
	"mlp.hook_pre_linear": "current.mlp.hook_pre", # NOTE: this is the W_in activation
	"mlp.hook_post": "current.mlp.hook_pre", # NOTE: this is silu(hook_pre) * hook_pre_linear
	"hook_mlp_out": None, 
	"hook_resid_post": None,
	"ln_final.hook_normalized": None,
}