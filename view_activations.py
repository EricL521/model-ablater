from pathlib import Path
import argparse
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
from helper_functions.shown_layers import SHOWN_LAYERS
from helper_functions.apply_mapping import apply_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
parser.add_argument('--show-values', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--mapping', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
model_id = args.model_id
show_values = args.show_values
do_mapping = args.mapping
local_dir = Path('.') / 'models' / model_id

print("model_id:", model_id)
print("show_values:", show_values)
print("do_mapping:", do_mapping)

# Show activations
ACTIVATIONS_FOLDER = local_dir / 'activations'

# Load activations and mappings if needed
activations = None
mappings = None
if show_values:
	try:
		activations = torch.load(local_dir / 'activations.pt', weights_only=False)
	except Exception as e:
		print(f"Warning: Could not load activations.pt: {e}")
if show_values and do_mapping:
	try:
		mappings = np.load(local_dir / 'mappings.npz', allow_pickle=True)
	except Exception as e:
		print(f"Warning: Could not load mappings.npz: {e}")

class LayerViewer:
	def __init__(self, root, activation_folder):
		self.root = root
		self.root.title("Token Activation Viewer")
		
		# Initial window size
		self.window_width = 800
		self.window_height = 600
		self.root.geometry(f"{self.window_width}x{self.window_height}")
		
		# Make window resizable
		self.root.resizable(True, True)
		
		# Zoom and pan state
		self.zoom_level = 1.0
		self.min_zoom = 0.05
		self.max_zoom = 75.0
		self.pan_x = 0
		self.pan_y = 0
		self.is_panning = False
		self.last_x = 0
		self.last_y = 0
		self.buffer = 1  # pixels of buffer
		
		# Image caching
		self.current_image = None
		self.current_image_path = None
		self.current_photo = None
		
		# Cache for image dimensions
		self.dimension_cache = {}  # (width, height) -> (zoom, pan_x, pan_y)
		
		# Get list of activation files
		self.activation_files = [f for f in os.listdir(activation_folder) if f.lower().endswith(('.png'))]
		# Sort files by token number
		self.activation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
		self.current_index = 0
		
		# Create main frame
		self.main_frame = ttk.Frame(root)
		self.main_frame.pack(fill=tk.BOTH, expand=True)
		
		# Add image title label at the top
		self.title_label = ttk.Label(self.main_frame, text="", font=("TkDefaultFont", 14, "bold"))
		self.title_label.pack(fill=tk.X, pady=(5, 0))
		
		# Create canvas for image display
		self.canvas = tk.Canvas(self.main_frame)
		self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		
		# Create button frame
		self.button_frame = ttk.Frame(self.main_frame)
		self.button_frame.pack(fill=tk.X, pady=5)
		
		# Create navigation buttons
		self.prev_button = ttk.Button(self.button_frame, text="Previous Token", command=self.show_previous)
		self.prev_button.pack(side=tk.LEFT, padx=5)
		self.prev_button.bind("<ButtonPress-1>", self.start_continuous_prev)
		self.prev_button.bind("<ButtonRelease-1>", self.stop_continuous_scroll)
		
		self.next_button = ttk.Button(self.button_frame, text="Next Token", command=self.show_next)
		self.next_button.pack(side=tk.RIGHT, padx=5)
		self.next_button.bind("<ButtonPress-1>", self.start_continuous_next)
		self.next_button.bind("<ButtonRelease-1>", self.stop_continuous_scroll)
		
		# Create status label
		self.status_label = ttk.Label(self.main_frame, text="")
		self.status_label.pack(fill=tk.X, pady=5)
		
		# Add pixel info label at the very bottom
		self.pixel_info_label = ttk.Label(self.main_frame, text="", anchor="w")
		self.pixel_info_label.pack(fill=tk.X, pady=(0, 5))
		
		# Bind mouse events
		self.canvas.bind("<MouseWheel>", self.zoom)  # Windows
		self.canvas.bind("<Button-4>", self.zoom)    # Linux scroll up
		self.canvas.bind("<Button-5>", self.zoom)    # Linux scroll down
		self.canvas.bind("<ButtonPress-1>", self.start_pan)
		self.canvas.bind("<ButtonRelease-1>", self.stop_pan)
		self.canvas.bind("<B1-Motion>", self.pan)
		self.canvas.bind("<Motion>", self.update_pixel_info)
		
		# Bind arrow keys
		self.root.bind("<Left>", lambda e: self.show_previous())
		self.root.bind("<Right>", lambda e: self.show_next())
		
		# Continuous scroll state
		self.continuous_scroll_active = False
		self.continuous_scroll_direction = None
		self.scroll_interval = 100  # milliseconds between scrolls
		self.initial_delay = 500  # milliseconds before starting continuous scroll
		self.continuous_scroll_timer = None
		
		# Wait for canvas to be ready before showing first activation
		self.root.update_idletasks()

		# Bind resize event
		self.canvas.bind("<Configure>", self.on_window_resize)

		self.show_values = show_values
		self.activations = activations
		self.mappings = mappings

		if self.activation_files:
			self.show_current_activation()
		else:
			self.status_label.config(text="No activation images found in the folder")
	
	def on_window_resize(self, event):
		canvas_width = self.canvas.winfo_width()
		canvas_height = self.canvas.winfo_height()

		# Initialize last canvas size if not present
		if not hasattr(self, 'last_canvas_width') or not hasattr(self, 'last_canvas_height'):
			self.last_canvas_width = canvas_width
			self.last_canvas_height = canvas_height

		if self.current_image and self.last_canvas_width > 0 and self.last_canvas_height > 0:
			# Calculate scale factors
			scale_x = canvas_width / self.last_canvas_width
			scale_y = canvas_height / self.last_canvas_height
			scale = (scale_x * scale_y) ** 0.5

			# Adjust pan_x and pan_y so the same image point stays at the center
			self.pan_x *= scale
			self.pan_y *= scale

	 		# Update zoom_level proportionally
			self.zoom_level *= scale

		# Update last canvas size
		self.last_canvas_width = canvas_width
		self.last_canvas_height = canvas_height
		
		self.show_current_activation()
	
	def start_pan(self, event):
		self.is_panning = True
		self.last_x = event.x
		self.last_y = event.y
	
	def stop_pan(self, event):
		self.is_panning = False
	
	def pan(self, event):
		if self.is_panning:
			dx = event.x - self.last_x
			dy = event.y - self.last_y
			self.pan_x += dx
			self.pan_y += dy
			self.last_x = event.x
			self.last_y = event.y
			self.show_current_activation()
	
	def zoom(self, event):
		# Get mouse position relative to canvas
		mouse_x = self.canvas.canvasx(event.x)
		mouse_y = self.canvas.canvasy(event.y)
		
		# Calculate the point under the mouse in image coordinates
		image_x = (mouse_x - self.pan_x) / self.zoom_level
		image_y = (mouse_y - self.pan_y) / self.zoom_level
		
		# Determine zoom direction
		if event.num == 5 or event.delta < 0:  # scroll down
			self.zoom_level = max(self.min_zoom, self.zoom_level * 0.9)
		elif event.num == 4 or event.delta > 0:  # scroll up
			self.zoom_level = min(self.max_zoom, self.zoom_level * 1.1)
		
		# Calculate new pan position to keep the point under the mouse
		self.pan_x = mouse_x - (image_x * self.zoom_level)
		self.pan_y = mouse_y - (image_y * self.zoom_level)
		
		self.show_current_activation()
	
	def show_current_activation(self):
		if not self.activation_files:
			return
			
		activation_path = os.path.join(ACTIVATIONS_FOLDER, self.activation_files[self.current_index])
	
		# Only load the image if it's different from the current one
		if activation_path != self.current_image_path:
			try:
				self.current_image = Image.open(activation_path)
				self.current_image_path = activation_path
				
				# Get image dimensions
				img_width, img_height = self.current_image.size
				dim_key = (img_width, img_height)
				
				# Check if we have cached settings for this dimension
				if dim_key in self.dimension_cache:
					# Restore cached settings
					self.zoom_level, self.pan_x, self.pan_y = self.dimension_cache[dim_key]
				else:
					# Calculate initial zoom to fit the image
					canvas_width = self.canvas.winfo_width()
					canvas_height = self.canvas.winfo_height()
					
					# Calculate scale factors for both dimensions
					scale_x = canvas_width / img_width
					scale_y = canvas_height / img_height
					
					# Use the smaller scale to ensure the image fits
					self.zoom_level = min(scale_x, scale_y) * 0.9  # 0.9 to leave some margin
					self.zoom_level = max(self.min_zoom, self.zoom_level)
					
					# Center the image
					self.pan_x = (canvas_width - (img_width * self.zoom_level)) / 2
					self.pan_y = (canvas_height - (img_height * self.zoom_level)) / 2
					
					# Cache the settings
					self.dimension_cache[dim_key] = (self.zoom_level, self.pan_x, self.pan_y)
			except Exception as e:
				self.status_label.config(text=f"Error loading activation: {str(e)}")
				return
		
		try:
	 		# Get token number
			self.token_num = int(self.activation_files[self.current_index].split('_')[1].split('.')[0])

			# Update image title at the top
			self.title_label.config(text=f"Token {self.token_num}")

			# Update status
			self.status_label.config(
				text=f"Token {self.token_num} of {len(self.activation_files) - 1} | Zoom: {self.zoom_level:.2f}x | Size: {self.current_image.width}x{self.current_image.height}"
			)
			
			# Update cache with current settings
			dim_key = (self.current_image.width, self.current_image.height)
			self.dimension_cache[dim_key] = (self.zoom_level, self.pan_x, self.pan_y)
	 
			# Clear canvas
			self.canvas.delete("all")
	 
			# Get visible area
			canvas_width = self.canvas.winfo_width()
			canvas_height = self.canvas.winfo_height()
			
			# Calculate visible region in original image coordinates with buffer
			visible_left = max(0, -self.pan_x / self.zoom_level - self.buffer)
			visible_top = max(0, -self.pan_y / self.zoom_level - self.buffer)
			visible_right = min(self.current_image.width, (canvas_width - self.pan_x) / self.zoom_level + self.buffer)
			visible_bottom = min(self.current_image.height, (canvas_height - self.pan_y) / self.zoom_level + self.buffer)
	 
			# Ensure coordinates are valid (left < right, top < bottom), if not, then don't show anything
			if visible_left >= visible_right or visible_top >= visible_bottom:
				return
			
			# Crop to visible region
			visible_region = self.current_image.crop((
				int(visible_left),
				int(visible_top),
				int(visible_right),
				int(visible_bottom)
			))
			
			# Calculate new size for visible region
			new_width = int(visible_region.width * self.zoom_level)
			new_height = int(visible_region.height * self.zoom_level)
	 
	 		# Calculate image position with buffer offset
			x = max(self.pan_x % self.zoom_level - self.zoom_level - int(self.buffer * self.zoom_level), self.pan_x)
			y = max(self.pan_y % self.zoom_level - self.zoom_level - int(self.buffer * self.zoom_level), self.pan_y)
			
			# Draw black border around the image
			border_size = 1
			self.canvas.create_rectangle(
				x - border_size, y - border_size,
				x + new_width + border_size, y + new_height + border_size,
				outline='#333333', width=border_size
			)
	 
	 		# If new width or height is 0, then only show the border
			if new_width == 0 or new_height == 0:
				return
			
			# Resize only the visible portion
			resized_region = visible_region.resize((new_width, new_height), Image.Resampling.NEAREST)
			
			# Convert to PhotoImage
			photo = ImageTk.PhotoImage(resized_region)
			
			# Display new image
			self.canvas.create_image(x, y, image=photo, anchor=tk.NW)
			self.canvas.image = photo  # Keep a reference
		except Exception as e:
			self.status_label.config(text=f"Error displaying activation: {str(e)}")
	
	def show_next(self):
		if self.activation_files:
			self.current_index = (self.current_index + 1) % len(self.activation_files)
			self.pan_x = 0  # Reset pan when changing images
			self.pan_y = 0
			self.show_current_activation()
	
	def show_previous(self):
		if self.activation_files:
			self.current_index = (self.current_index - 1) % len(self.activation_files)
			self.pan_x = 0  # Reset pan when changing images
			self.pan_y = 0
			self.show_current_activation()
	
	def start_continuous_prev(self, event):
		self.continuous_scroll_direction = "prev"
		# Schedule continuous scroll after delay
		self.continuous_scroll_timer = self.root.after(self.initial_delay, self.start_continuous_scroll)
	
	def start_continuous_next(self, event):
		self.continuous_scroll_direction = "next"
		# Schedule continuous scroll after delay
		self.continuous_scroll_timer = self.root.after(self.initial_delay, self.start_continuous_scroll)
	
	def start_continuous_scroll(self):
		self.continuous_scroll_active = True
		self.continuous_scroll()
	
	def stop_continuous_scroll(self, event):
		self.continuous_scroll_active = False
		# Cancel the initial delay timer if it's still pending
		if self.continuous_scroll_timer:
			self.root.after_cancel(self.continuous_scroll_timer)
			self.continuous_scroll_timer = None
	
	def continuous_scroll(self):
		if not self.continuous_scroll_active:
			return
		
		if self.continuous_scroll_direction == "prev":
			self.show_previous()
		else:
			self.show_next()
		
		# Schedule next scroll
		self.root.after(self.scroll_interval, self.continuous_scroll)

	def update_pixel_info(self, event):
		if not self.current_image or not self.show_values:
			self.pixel_info_label.config(text="")
			return
		# Get mouse position relative to canvas
		mouse_x = self.canvas.canvasx(event.x)
		mouse_y = self.canvas.canvasy(event.y)
		# Convert to image coordinates
		image_x = int(np.floor((mouse_x - self.pan_x) / self.zoom_level))
		image_y = int(np.floor((mouse_y - self.pan_y) / self.zoom_level))
		# Get image dimensions
		image_height = self.current_image.height
		# Clamp to image bounds
		if 0 <= image_x < self.current_image.width and 0 <= image_y < image_height:
			info = ""
			try:
				# Calculate which layer we're in based on x position
				# Each layer takes up its width + 1 pixel for spacing
				layer_width = 0
				current_layer = None
				
				for key in self.activations.keys():
					layer_num_array = key.split('.')[:2:]
					layer_name = '.'.join(key.split('.')[2::])
					if not (layer_name in SHOWN_LAYERS or key in SHOWN_LAYERS):
						continue
					# Get activation shape for this layer
					act = self.activations[key]
					act_np = act.detach().cpu().numpy()
					if act_np.shape[0] == 1:
						act_np = act_np[0]
					
					# Calculate width of this layer's visualization
					if act_np.ndim == 2:
						# 2D: (tokens, d_model)
						# Width is d_model, which is the second dimension
						layer_width += act_np.shape[1] // image_height
					elif act_np.ndim == 3 and act_np.shape[1] == len(self.activation_files) and act_np.shape[2] == len(self.activation_files):
						# 3D: (heads, tokens, tokens)
						layer_width += act_np.shape[1]
					elif act_np.ndim == 3:
						# 3D: (tokens, heads, d_head)
						# Width is d_head, which is the third dimension
						layer_width += act_np.shape[2] // image_height * act_np.shape[1]
					
					# Add spacing
					layer_width += 1
					
					if image_x < layer_width:
						current_layer = key
						break
				
				if current_layer is not None:
					info = f"Layer: {current_layer}"
					# Get activation for this layer
					act = self.activations[current_layer]
					act_np = act.detach().cpu().numpy().copy()
					if act_np.shape[0] == 1:
						act_np = act_np[0]
					if do_mapping:
						mapping_id = None
						shown_layer_key = key if key in SHOWN_LAYERS else layer_name
						if isinstance(SHOWN_LAYERS[shown_layer_key], str):
							mapping_id = SHOWN_LAYERS[shown_layer_key]\
								.replace('current', '.'.join(layer_num_array))\
								.replace('prev', '.'.join([layer_num_array[0], str(int(layer_num_array[1]) - 1)]))
						mapping = self.mappings.get(mapping_id, None)
						if mapping is not None:
							for i in range(act_np.shape[0]):
								act_np[i] = apply_mapping(act_np[i], mapping)
     
					# Calculate position within the layer
					layer_start_x = layer_width - 1
					if act_np.ndim == 2:
						layer_start_x -= act_np.shape[1] // image_height
						relative_x = image_x - layer_start_x
      
						if relative_x == act_np.shape[1] // image_height:
							info = " | ".join([info, "(spacing)"])
						else:
							# 2D: (tokens, d_model)
							position = (self.token_num, relative_x * image_height + image_y)
							value = act_np[position]
							info = " | ".join([info, f"Position: {position}",  f"Value: {value:.4f}"])
					elif act_np.ndim == 3 and act_np.shape[1] == len(self.activation_files) and act_np.shape[2] == len(self.activation_files):
						layer_start_x -= act_np.shape[1]
						layer_start_y = (image_height - act_np.shape[0]) // 2
						relative_x = image_x - layer_start_x
						relative_y = image_y - layer_start_y
						if relative_x == act_np.shape[1] or relative_y < 0 or relative_y >= act_np.shape[0]:
							info = " | ".join([info, "(spacing)"])
						else:
							# 3D: (heads, tokens, tokens)
							position = (relative_y, self.token_num, relative_x)
							value = act_np[position]
							info = " | ".join([info, f"Position: {position}",  f"Value: {value:.4f}"])
					elif act_np.ndim == 3:
						layer_start_x -= act_np.shape[2] // image_height * act_np.shape[1]
						relative_x = image_x - layer_start_x
      
						if relative_x == act_np.shape[2] // image_height * act_np.shape[1]:
							info = " | ".join([info, "(spacing)"])
						else:
							# 3D: (tokens, heads, d_head)
							head_idx = relative_x // (act_np.shape[2] // image_height)
							d_head_idx = (image_y * (act_np.shape[2] // image_height)) + (relative_x % (act_np.shape[2] // image_height))
							position = (self.token_num, head_idx, d_head_idx)
							value = act_np[position]
							info = " | ".join([info, f"Position: {position}",  f"Value: {value:.4f}"])
			except Exception as e:
				info = f"Error: {e}"
			self.pixel_info_label.config(text=info)
		else:
			self.pixel_info_label.config(text="")

if __name__ == "__main__":
	root = tk.Tk()
	app = LayerViewer(root, ACTIVATIONS_FOLDER)
	root.mainloop()
