from pathlib import Path
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
args = parser.parse_args()
model_id = args.model_id
local_dir = Path('.') / 'models' / model_id

print("model_id:", model_id)

# Show layers
LAYER_FOLDER = local_dir / 'layers'

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
		
		# Bind mouse events
		self.canvas.bind("<MouseWheel>", self.zoom)  # Windows
		self.canvas.bind("<Button-4>", self.zoom)    # Linux scroll up
		self.canvas.bind("<Button-5>", self.zoom)    # Linux scroll down
		self.canvas.bind("<ButtonPress-1>", self.start_pan)
		self.canvas.bind("<ButtonRelease-1>", self.stop_pan)
		self.canvas.bind("<B1-Motion>", self.pan)
		
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
		self.root.bind("<Configure>", self.on_window_resize)
  
		if self.layer_files:
			self.show_current_layer()
		else:
			self.status_label.config(text="No layers found in the folder")
	
	def on_window_resize(self, event):
		# Only handle main window resize events
		if event.widget == self.root:
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
			
			# Update status
			self.status_label.config(
				text=f"Layer {self.current_index + 1} of {len(self.layer_files)} | {layer_name} | Zoom: {self.zoom_level:.2f}x | Size: {self.current_image.width}x{self.current_image.height}"
			)
			
			# Update cache with current settings
			dim_key = (self.current_image.width, self.current_image.height)
			self.dimension_cache[dim_key] = (self.zoom_level, self.pan_x, self.pan_y)
   
			# Clear canvas
			self.canvas.delete("all")
   
			# Get visible area
			canvas_width = self.canvas.winfo_width()
			canvas_height = self.canvas.winfo_height()
			
			# Add buffer zone to prevent edge artifacts
			buffer = 1  # pixels of buffer per zoom level
			
			# Calculate visible region in original image coordinates with buffer
			visible_left = max(0, -self.pan_x / self.zoom_level - buffer)
			visible_top = max(0, -self.pan_y / self.zoom_level - buffer)
			visible_right = min(self.current_image.width, (canvas_width - self.pan_x) / self.zoom_level + buffer)
			visible_bottom = min(self.current_image.height, (canvas_height - self.pan_y) / self.zoom_level + buffer)
   
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
			x = max(self.pan_x % self.zoom_level - int(buffer * self.zoom_level), self.pan_x + int(buffer * self.zoom_level))
			y = max(self.pan_y % self.zoom_level - int(buffer * self.zoom_level), self.pan_y + int(buffer * self.zoom_level))
			
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

if __name__ == "__main__":
	root = tk.Tk()
	app = LayerViewer(root, LAYER_FOLDER)
	root.mainloop()
