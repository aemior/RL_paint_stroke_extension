import sys
import copy

import numpy as np

from Renders.RenderMath import *

sys.path.append('mypaint/mypaint')
from lib import surface, tiledsurface, brush, pixbufsurface, floodfill

class MyPainterAPI(object):
	actions_to_idx = {
        'pressure': 0,
        'size': 1,
        'control_x': 2,
        'control_y': 3,
        'end_x': 4,
        'end_y': 5,
        'start_x': 6,
        'start_y': 7,
        'entry_pressure': 8,
        'color_r': 9,
        'color_g': 10,
        'color_b': 11,
    }
	head = 0.25
	tail = 0.75

	def __init__(self, ScreenSize, BrushPath):
		self.screen_size = ScreenSize
		self.brush_path = BrushPath
		self.set_brush()
		self.brush = self.get_brush()

	def draw(self, action, brush, canvas, dtime=0.01):
		canvas.begin_atomic()

		s_x, s_y = action[self.actions_to_idx['start_x']]*self.screen_size, action[self.actions_to_idx['start_y']]*self.screen_size  
		e_x, e_y = action[self.actions_to_idx['end_x']]*self.screen_size, action[self.actions_to_idx['end_y']]*self.screen_size
		c_x, c_y = action[self.actions_to_idx['control_x']]*self.screen_size, action[self.actions_to_idx['control_y']]*self.screen_size
		color = (
            action[self.actions_to_idx['color_r']],
            action[self.actions_to_idx['color_g']],
            action[self.actions_to_idx['color_b']],
        )
		pressure = action[self.actions_to_idx['pressure']]
		entry_pressure = action[self.actions_to_idx['entry_pressure']]
		size = action[self.actions_to_idx['size']] * 4.0

		brush.brushinfo.set_color_rgb(color)
		brush.brushinfo.set_base_value('radius_logarithmic', size)

		self._stroke_to(brush, canvas, s_x, s_y, 0, 1)
		self._draw(brush, canvas, s_x, s_y, e_x, e_y, c_x, c_y, entry_pressure, pressure, size, color, dtime)
	
	def _stroke_to(self, b, s, x, y, pressure, duration=0.01):
		b.stroke_to(
                s.backend,
                x, y,
                pressure,
                0.0, 0.0,
                duration,0,0,0)
		s.end_atomic()
		s.begin_atomic()
	
	def _draw(self, b, s, s_x, s_y, e_x, e_y, c_x, c_y,
              entry_pressure, pressure, size, color, dtime):

		# if straight line or jump
		if pressure == 0:
			b.stroke_to(
					s.backend, e_x, e_y, pressure, 0, 0, dtime,0,0,0)
		else:
			self.curve(b, s, c_x, c_y, s_x, s_y, e_x, e_y, entry_pressure, pressure)
			
		# Relieve brush pressure for next jump
		self._stroke_to(b, s, e_x, e_y, 0)

		s.end_atomic()
		s.begin_atomic()

	def curve(self, b, s, cx, cy, sx, sy, ex, ey, entry_pressure, pressure):
		#entry_p, midpoint_p, junk, prange2, head, tail
		entry_p, midpoint_p, prange1, prange2, h, t = \
				self._line_settings(entry_pressure, pressure)

		points_in_curve = 100
		mx, my = midpoint(sx, sy, ex, ey)
		length, nx, ny = length_and_normal(mx, my, cx, cy)
		cx, cy = multiply_add(mx, my, nx, ny, length*2)
		x1, y1 = difference(sx, sy, cx, cy)
		x2, y2 = difference(cx, cy, ex, ey)
		head = points_in_curve * h
		head_range = int(head)+1
		tail = points_in_curve * t
		tail_range = int(tail)+1
		tail_length = points_in_curve - tail

		# Beginning
		px, py = point_on_curve_1(1, cx, cy, sx, sy, x1, y1, x2, y2)
		length, nx, ny = length_and_normal(sx, sy, px, py)
		bx, by = multiply_add(sx, sy, nx, ny, 0.25)
		self._stroke_to(b, s, bx, by, entry_p)
		pressure = abs(1/head * prange1 + entry_p)
		self._stroke_to(b, s, px, py, pressure)

		for i in range(2, head_range):
			px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
			pressure = abs(i/head * prange1 + entry_p)
			self._stroke_to(b, s, px, py, pressure)

		# Middle
		for i in range(head_range, tail_range):
			px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
			self._stroke_to(b, s, px, py, midpoint_p)

		# End
		for i in range(tail_range, points_in_curve+1):
			px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
			pressure = abs((i-tail)/tail_length * prange2 + midpoint_p)
			self._stroke_to(b, s, px, py, pressure)

		return pressure

	def _line_settings(self, entry_pressure, pressure):
		p1 = entry_pressure
		p2 = (entry_pressure + pressure) / 2
		p3 = pressure
		if self.head == 0.0001:
			p1 = p2
		prange1 = p2 - p1
		prange2 = p3 - p2
		return p1, p2, prange1, prange2, self.head, self.tail

	def set_brush(self):
		with open(self.brush_path) as fp:
			self.binfo = brush.BrushInfo(fp.read())
		
	def get_brush(self):
		return brush.Brush(self.binfo)

	def get_surface(self):
		return tiledsurface.Surface()

	def get_image(self, canvas, size = None):
		canvas.end_atomic()
		if size is None:
			scanline_strips = surface.scanline_strips_iter(canvas, (0,0,self.screen_size, self.screen_size), alpha=True)
		else:
			scanline_strips = surface.scanline_strips_iter(canvas, (0,0,size[0], size[1]), alpha=True)
		patchs = [copy.deepcopy(i) for i in scanline_strips]
		array = np.concatenate(tuple(patchs),axis=0)
		return array