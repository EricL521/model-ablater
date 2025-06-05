from pathlib import Path
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import re
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
parser.add_argument('--show-activation', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
model_id = args.model_id
show_activation = args.show_activation
local_dir = Path('.') / 'models' / model_id

print("model_id:", model_id)

# Show layers
LAYER_FOLDER = local_dir / 'layers'

# Load activations if needed
activations = None
if show_activation:
	try:
		activations = torch.load(local_dir / 'activations.pt', weights_only=False)
	except Exception as e:
		print(f"Warning: Could not load activations.pt: {e}")

class LayerViewer:
	def __init__(self, root, layer_folder):
		self.root = root
		self.root.title("Layer Viewer")
		
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
		
		# Get list of layer files
		self.layer_files = [f for f in os.listdir(layer_folder) if f.lower().endswith(('.png'))]
		# Sort files alphanumerically
		self.layer_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', x)])
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
		self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.show_previous)
		self.prev_button.pack(side=tk.LEFT, padx=5)
		self.prev_button.bind("<ButtonPress-1>", self.start_continuous_prev)
		self.prev_button.bind("<ButtonRelease-1>", self.stop_continuous_scroll)
		
		self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next)
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
		
		# Wait for canvas to be ready before showing first layer
		self.root.update_idletasks()

		# Bind resize event
		self.canvas.bind("<Configure>", self.on_window_resize)

		self.show_activation = show_activation
		self.activations = activations

		if self.layer_files:
			self.show_current_layer()
		else:
			self.status_label.config(text="No layers found in the folder")
	
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
		
		self.show_current_layer()
	
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
			self.show_current_layer()
	
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
		
		self.show_current_layer()
	
	def show_current_layer(self):
		if not self.layer_files:
			return
			
		layer_path = os.path.join(LAYER_FOLDER, self.layer_files[self.current_index])
	
		# Only load the image if it's different from the current one
		if layer_path != self.current_image_path:
			try:
				self.current_image = Image.open(layer_path)
				self.current_image_path = layer_path
				
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
				self.status_label.config(text=f"Error loading layer: {str(e)}")
				return
		
		try:
	 		# Get layer name (remove number prefix)
			layer_name = self.layer_files[self.current_index]
			if '_' in layer_name:
				layer_name = layer_name.split('_', 1)[1]  # Remove everything before first underscore
			layer_name = os.path.splitext(layer_name)[0]  # Remove extension

			# Update image title at the top
			self.title_label.config(text=layer_name)

			# Update status
			self.status_label.config(
				text=f"Layer {self.current_index + 1} of {len(self.layer_files)} | Zoom: {self.zoom_level:.2f}x | Size: {self.current_image.width}x{self.current_image.height}"
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
			self.status_label.config(text=f"Error displaying layer: {str(e)}")
	
	def show_next(self):
		if self.layer_files:
			self.current_index = (self.current_index + 1) % len(self.layer_files)
			self.pan_x = 0  # Reset pan when changing images
			self.pan_y = 0
			self.show_current_layer()
	
	def show_previous(self):
		if self.layer_files:
			self.current_index = (self.current_index - 1) % len(self.layer_files)
			self.pan_x = 0  # Reset pan when changing images
			self.pan_y = 0
			self.show_current_layer()
	
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
		if not self.current_image:
			self.pixel_info_label.config(text="")
			return
		# Get mouse position relative to canvas
		mouse_x = self.canvas.canvasx(event.x)
		mouse_y = self.canvas.canvasy(event.y)
		# Convert to image coordinates
		image_x = int(np.floor((mouse_x - self.pan_x) / self.zoom_level))
		image_y = int(np.floor((mouse_y - self.pan_y) / self.zoom_level))
		# Clamp to image bounds
		if 0 <= image_x < self.current_image.width and 0 <= image_y < self.current_image.height:
			info = ""
			if self.show_activation and self.activations is not None:
				try:
					layer_file = self.layer_files[self.current_index]
					idx_and_key = os.path.splitext(layer_file)[0]
					idx, key = idx_and_key.split('_', 1)
					act = self.activations[key]
					act_np = act.detach().cpu().numpy()
					if act_np.shape[0] == 1:
						act_np = act_np[0]
					value = None
					is_spacing_row = False
					index_tuple = None
					if act_np.ndim == 2:
						# 2D: (tokens, d_model)
						n_rows = act_np.shape[0]
						if image_y % 2 == 1:
							is_spacing_row = True
							index_tuple = "(spacing)"
						else:
							act_row = image_y // 2
							if act_row < act_np.shape[0] and image_x < act_np.shape[1]:
								value = act_np[act_row, image_x]
							index_tuple = f"({act_row}, {image_x})"
					elif act_np.ndim == 3:
						# 3D: (tokens, heads, d_head)
						t, h, d = act_np.shape
						block = h
						block_idx = image_y // (block + 1)
						in_block_idx = image_y % (block + 1)
						if in_block_idx == block:
							is_spacing_row = True
							index_tuple = "(spacing)"
						else:
							act_row = block_idx
							act_head = in_block_idx
							if act_row < t and act_head < h and image_x < d:
								value = act_np[act_row, act_head, image_x]
							index_tuple = f"({act_row}, {act_head}, {image_x})"
					if is_spacing_row:
						info = f"{index_tuple} | (spacing row)"
					elif value is not None:
						info = f"{index_tuple} | Value: {value:.4f}"
					else:
						info = f"{index_tuple} | Value: N/A"
				except Exception as e:
					info = f"Error: {e}"
			self.pixel_info_label.config(text=info)
		else:
			self.pixel_info_label.config(text="")

if __name__ == "__main__":
	root = tk.Tk()
	app = LayerViewer(root, LAYER_FOLDER)
	root.mainloop()
