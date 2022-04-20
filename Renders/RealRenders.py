import importlib
import numpy as np

from Renders.MyPaintStroke import MyPainterAPI
from Renders.WaterColorStroke import draw
from Renders.SNPStrokes import Renderer

class MyPaintStroke(object):
	def __init__(self, CanvasWidth, BrushType):
		self.action_size = 12
		self.canvas_width = CanvasWidth
		if BrushType == 'WaterInk':
			self.MPA = MyPainterAPI(CanvasWidth, 'mypaint/brushes/watercolor-02-paint.myb')
		elif BrushType == 'Pencil':
			self.MPA = MyPainterAPI(CanvasWidth, 'mypaint/brushes/2B_pencil.myb')
		elif BrushType == 'Charcoal':
			self.MPA = MyPainterAPI(CanvasWidth, 'mypaint/brushes/charcoal.myb')

	def SingleStroke(self, action, canvas=None):
		s = self.MPA.get_surface()
		self.MPA.draw(action.astype(np.float64), self.MPA.brush, s)
		image = self.MPA.get_image(s)
		if canvas is None:
			return image
		else:
			alpha = image[:,:,-1].astype(np.float32).reshape(128,128,1)/255
			canvas = canvas * (1-alpha) + image[:,:,:-1] * alpha
			return canvas.astype(np.uint8)

class WaterColorStroke(object):
	def __init__(self, cw=128):
		self.action_size = 13
		self.canvas_width = cw
		self.draw = draw
	def SingleStroke(self, action):
		res = np.ones((self.canvas_width, self.canvas_width, 4), np.float32)
		res[:,:,3] = 1 - self.draw(action[:10], self.canvas_width)
		res[:,:,:3] *= action[10:] 
		return (res*255).astype(np.uint8)

class SNPStroke(object):
	def __init__(self, cw=128, BrushType='oilpaintbrush'):
		if BrushType=='oilpaintbrush':
			self.action_size = 12
		elif BrushType=='markerpen':
			self.action_size = 12
		elif BrushType=='rectangle':
			self.action_size = 9
		self.canvas_width = cw
		self.rd = Renderer(CANVAS_WIDTH=self.canvas_width, renderer=BrushType) 
	def SingleStroke(self, action):
		res = np.ones((self.canvas_width, self.canvas_width, 4), np.uint8)
		res[:,:,:3], res[:,:,3] = self.rd.draw_stroke(action)
		return res
		