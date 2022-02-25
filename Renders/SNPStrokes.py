import random

import numpy as np
import cv2

def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)


class Renderer():

    def __init__(self, renderer='oilpaintbrush', CANVAS_WIDTH=128, train=True, canvas_color='black'):

        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.renderer = renderer
        self.stroke_params = None
        self.canvas_color = canvas_color

        self.canvas = None
        self.create_empty_canvas()

        self.train = train

        if self.renderer in ['markerpen']:
            self.d = 12 # x0, y0, x1, y1, x2, y2, A, radius0, radius2, R, G, B
            self.d_shape = 8
            self.d_color = 3
            self.d_alpha = 1
        elif self.renderer in ['oilpaintbrush']:
            self.d = 12 # xc, yc, w, h, theta, A, R0, G0, B0, R2, G2, B2
            self.d_shape = 5
            self.d_color = 6
            self.d_alpha = 1
            self.brush_small_vertical = cv2.imread(
                r'./Renders/StrokeTemplate/brush_fromweb2_small_vertical.png', cv2.IMREAD_GRAYSCALE)
            self.brush_small_horizontal = cv2.imread(
                r'./Renders/StrokeTemplate/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_vertical = cv2.imread(
                r'./Renders/StrokeTemplate/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_horizontal = cv2.imread(
                r'./Renders/StrokeTemplate/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)
        elif self.renderer in ['rectangle']:
            self.d = 9 # xc, yc, w, h, theta, A, R, G, B
            self.d_shape = 5
            self.d_color = 3
            self.d_alpha = 1
        else:
            raise NotImplementedError(
                'Wrong renderer name %s (choose one from [watercolor, markerpen, oilpaintbrush, rectangle] ...)'
                % self.renderer)

    def create_empty_canvas(self):
        if self.canvas_color == 'white':
            self.canvas = np.ones(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')
        else:
            self.canvas = np.zeros(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')



    def check_stroke(self):
        r_ = 1.0
        if self.renderer in ['markerpen', 'watercolor']:
            r_ = max(self.stroke_params[6], self.stroke_params[7])
        elif self.renderer in ['oilpaintbrush']:
            r_ = max(self.stroke_params[2], self.stroke_params[3])
        elif self.renderer in ['rectangle']:
            r_ = max(self.stroke_params[2], self.stroke_params[3])
        if r_ > 0.025:
            return True
        else:
            return False


    def draw_stroke(self, action):

        if self.renderer == 'markerpen':
            return self._draw_markerpen(action)
        elif self.renderer == 'oilpaintbrush':
            return self._draw_oilpaintbrush(action)
        elif self.renderer == 'rectangle':
            return self._draw_rectangle(action)



    def _draw_rectangle(self, action):
        
        self.stroke_params = action

        # xc, yc, w, h, theta, A, R, G, B
        x0, y0, w, h, theta, A = self.stroke_params[0:6]
        R0, G0, B0 = self.stroke_params[6:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH // 4)
        h = (int)(1 + h * self.CANVAS_WIDTH // 4)
        theta = np.pi*theta
        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing

        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        ptc = (x0, y0)
        pt0 = rotate_pt(pt=(x0 - w, y0 - h), rotate_center=ptc, theta=theta)
        pt1 = rotate_pt(pt=(x0 + w, y0 - h), rotate_center=ptc, theta=theta)
        pt2 = rotate_pt(pt=(x0 + w, y0 + h), rotate_center=ptc, theta=theta)
        pt3 = rotate_pt(pt=(x0 - w, y0 + h), rotate_center=ptc, theta=theta)

        ppt = np.array([pt0, pt1, pt2, pt3], np.int32)
        ppt = ppt.reshape((-1, 1, 2))
        cv2.fillPoly(self.foreground, [ppt], color, lineType=cv2.LINE_AA)
        cv2.fillPoly(self.stroke_alpha_map, [ppt], alpha, lineType=cv2.LINE_AA)

        return self.foreground, (A * self.stroke_alpha_map[:,:,0].astype(np.float32)).astype(np.uint8)

    def _draw_markerpen(self, action):

        self.stroke_params = action

        # x0, y0, x1, y1, x2, y2, radius0, radius2, A, R, G, B
        x0, y0, x1, y1, x2, y2, radius, _, A = self.stroke_params[0:9]
        R0, G0, B0 = self.stroke_params[9:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        x1 = _normalize(x1, self.CANVAS_WIDTH)
        x2 = _normalize(x2, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        y1 = _normalize(y1, self.CANVAS_WIDTH)
        y2 = _normalize(y2, self.CANVAS_WIDTH)
        radius = (int)(1 + radius * self.CANVAS_WIDTH // 4)

        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing

        if abs(x0-x2) + abs(y0-y2) < 4: # too small, dont draw
            self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
            self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
            return self.foreground, (A * self.stroke_alpha_map[:,:,0].astype(np.float32)).astype(np.uint8)
            
        
        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2
            y = (1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2

            ptc = (x, y)
            dx = 2 * (t - 1) * x0 + 2 * (1 - 2 * t) * x1 + 2 * t * x2
            dy = 2 * (t - 1) * y0 + 2 * (1 - 2 * t) * y1 + 2 * t * y2

            theta = np.arctan2(dx, dy) - np.pi/2
            pt0 = rotate_pt(pt=(x - radius, y - radius), rotate_center=ptc, theta=theta)
            pt1 = rotate_pt(pt=(x + radius, y - radius), rotate_center=ptc, theta=theta)
            pt2 = rotate_pt(pt=(x + radius, y + radius), rotate_center=ptc, theta=theta)
            pt3 = rotate_pt(pt=(x - radius, y + radius), rotate_center=ptc, theta=theta)
            ppt = np.array([pt0, pt1, pt2, pt3], np.int32)
            ppt = ppt.reshape((-1, 1, 2))
            cv2.fillPoly(self.foreground, [ppt], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(self.stroke_alpha_map, [ppt], alpha, lineType=cv2.LINE_AA)

        return self.foreground, (A * self.stroke_alpha_map[:,:,0].astype(np.float32)).astype(np.uint8)


    def _draw_oilpaintbrush(self, action):

        self.stroke_params = action

        # xc, yc, w, h, theta, A, R0, G0, B0, R2, G2, B2
        x0, y0, w, h, theta, A = self.stroke_params[0:6]
        R0, G0, B0, R2, G2, B2 = self.stroke_params[6:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH)
        h = (int)(1 + h * self.CANVAS_WIDTH)
        theta = np.pi*theta

        if w * h / (self.CANVAS_WIDTH**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal
        self.foreground, self.stroke_alpha_map = create_transformed_brush(
            brush, self.CANVAS_WIDTH, self.CANVAS_WIDTH,
            x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

        return self.foreground, (A * self.stroke_alpha_map[:,:,0].astype(np.float32)).astype(np.uint8)


def create_transformed_brush(brush, canvas_w, canvas_h,
                      x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2):

    brush_alpha = np.stack([brush, brush, brush], axis=-1)
    brush_alpha = (brush_alpha > 0).astype(np.float32)
    brush_alpha = (brush_alpha*255).astype(np.uint8)
    colormap = np.zeros([brush.shape[0], brush.shape[1], 3], np.float32)
    for ii in range(brush.shape[0]):
        t = ii / brush.shape[0]
        this_color = [(1 - t) * R0 + t * R2,
                      (1 - t) * G0 + t * G2,
                      (1 - t) * B0 + t * B2]
        colormap[ii, :, :] = np.expand_dims(this_color, axis=0)

    brush = np.expand_dims(brush, axis=-1).astype(np.float32) / 255.
    brush = (brush * colormap * 255).astype(np.uint8)
    # plt.imshow(brush), plt.show()

    M1 = build_transformation_matrix([-brush.shape[1]/2, -brush.shape[0]/2, 0])
    M2 = build_scale_matrix(sx=w/brush.shape[1], sy=h/brush.shape[0])
    M3 = build_transformation_matrix([0,0,theta])
    M4 = build_transformation_matrix([x0, y0, 0])

    M = update_transformation_matrix(M1, M2)
    M = update_transformation_matrix(M, M3)
    M = update_transformation_matrix(M, M4)

    brush = cv2.warpAffine(
        brush, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    brush_alpha = cv2.warpAffine(
        brush_alpha, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

    return brush, brush_alpha


def build_scale_matrix(sx, sy):
    transform_matrix = np.zeros((2, 3))
    transform_matrix[0, 0] = sx
    transform_matrix[1, 1] = sy
    return transform_matrix


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]
#

def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix

def rotate_pt(pt, rotate_center, theta, return_int=True):

    # theta in [0, pi]
    x, y = pt[0], pt[1]
    xc, yc = rotate_center[0], rotate_center[1]

    x_ = (x-xc) * np.cos(theta) + (y-yc) * np.sin(theta) + xc
    y_ = -1 * (x-xc) * np.sin(theta) + (y-yc) * np.cos(theta) + yc

    if return_int:
        x_, y_ = int(x_), int(y_)

    pt_ = (x_, y_)

    return pt_