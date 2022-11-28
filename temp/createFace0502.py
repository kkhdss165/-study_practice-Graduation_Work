import bpy
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

ORG_JAMLINE_POINTS =[(84, 341), (93, 404), (104, 466), (119, 525),
                 (139, 583), (173, 632), (219, 672), (271, 704),
                 (330, 714),
                 (387, 704), (436, 672), (477, 631),(505, 580),
                 (521, 521), (532, 462), (544, 401), (549, 341)]

ORG_RIGHT_EYEBROW_POINTS = [(142, 292), (176, 267), (219, 262), (263, 268), (306, 282)]
ORG_LEFT_EYEBROW_POINTS = [(375, 282), (413, 265), (453, 258), (493, 261), (522, 289)]
ORG_MOUSE_OUTLINE_POINTS = [(257, 573), (289, 551), (318, 535), (341, 542), (361, 534), (386, 547), (410, 570), (386, 595), (362, 607), (340, 610), (315, 608), (287, 596)]

ORG_RIGHT_EYE_POINTS = [(191, 339), (217, 325), (246, 325), (269, 342), (242, 347), (215, 347)]
ORG_LEFT_EYE_POINTS = [(403, 340), (426, 323), (454, 322), (478, 338), (455, 346), (427, 345)]
ORG_NOSE_POINTS = [(337, 326), (339, 365), (341, 403), (344, 443), (300, 483), (320, 486), (340, 491), (360, 485), (379, 480)]

NEW_JAMLINE_POINTS= copy.deepcopy(ORG_JAMLINE_POINTS)

def print_point(Point):
    X=[]
    Y=[]
    for i in Point:
        X.append(i[0])
        Y.append(-i[1])
    plt.plot(X,Y)
    plt.show()
    print(Point)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

#좌표 가공(좌우 대칭)
Radius = abs(ORG_JAMLINE_POINTS[0][0] - ORG_JAMLINE_POINTS[16][0])/2
mid_x = (ORG_JAMLINE_POINTS[0][0] + ORG_JAMLINE_POINTS[16][0] + ORG_JAMLINE_POINTS[8][0])/ 3
for i in range(int(len(ORG_JAMLINE_POINTS)/2)):
    len_x = (ORG_JAMLINE_POINTS[-(i+1)][0]-ORG_JAMLINE_POINTS[i][0])/2
    mid_y = (ORG_JAMLINE_POINTS[-(i+1)][1]+ORG_JAMLINE_POINTS[i][1])/2

    NEW_JAMLINE_POINTS[i] = (mid_x-len_x, mid_y)
    NEW_JAMLINE_POINTS[-(i+1)] = (mid_x+len_x, mid_y)

    print(NEW_JAMLINE_POINTS[i][0]+NEW_JAMLINE_POINTS[-(i+1)][0])

NEW_JAMLINE_POINTS[8] = (mid_x,ORG_JAMLINE_POINTS[8][1])
Height = NEW_JAMLINE_POINTS[8][1] - NEW_JAMLINE_POINTS[0][1]

center = (NEW_JAMLINE_POINTS[8][0],NEW_JAMLINE_POINTS[0][1])
print(Height, Radius)
print(center)

R_94 = Radius * 94 / 144.6
R_110 = Radius * 110/144.6
Line_X = (R_110**2 + Height**2)**0.5
Line_Z = (Line_X**2 - Radius**2)**0.5
sinA = Height/Line_X
cosA = R_110/Line_X
sinB = Line_Z/Line_X
cosB = Radius/Line_X
Line_Y = Radius/(sinA * cosB + sinB * cosA)

#1번 평면 좌표 가공
flat1_point = copy.deepcopy(NEW_JAMLINE_POINTS)
temp_list=[]
r1 = NEW_JAMLINE_POINTS[8][0] - R_94
r2 = NEW_JAMLINE_POINTS[8][0]
w = r2-r1
for i in range(9):
#    new_point.append((r1 + (1 - (6-i)*(6-i)/36) * w, NEW_JAMLINE_POINTS[i][1]))
#    new_point.append((r1 + (1 - i*i / 36) * w, NEW_JAMLINE_POINTS[i][1]))
    temp_list.append((r1 + w * i**2 / 64, NEW_JAMLINE_POINTS[i][1]))

for i in range(9):
    flat1_point[i] = (temp_list[i][0], NEW_JAMLINE_POINTS[i][1])
    flat1_point[-(i+1)] = (2*center[0] - temp_list[i][0], NEW_JAMLINE_POINTS[-(i+1)][1])

# 평면 생성 및 좌표 가공
temp_Point = []
for i in flat1_point:
    temp_Point.append(((i[0]-center[0])/R_94, (i[1]-center[1])/R_94))
    
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 94 / 94, enter_editmode=False, location=(0, 0, 94/110))
obj = bpy.data.objects[-1]
obj.name ="face001"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face001']
#select vertex
obj = bpy.context.active_object

for i in range(8,25):
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    diff_x = temp_Point[i-8][0] - vertex_list[i][0]
    diff_y = -temp_Point[i-8][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode = 'OBJECT')

#3번 평면 좌표 가공
xy_coord=[[NEW_JAMLINE_POINTS[0][0],NEW_JAMLINE_POINTS[0][1]], [center[0],center[1]+Line_Y], [NEW_JAMLINE_POINTS[16][0],NEW_JAMLINE_POINTS[16][1]]]
xy = np.array(xy_coord)
x =[]
for i in NEW_JAMLINE_POINTS:
    x.append(i[0])

print(x)
intrp = interp1d(xy[:,0], xy[:,1],  kind='quadratic')
y = intrp(x)
print(y)

# 평면 생성 및 좌표 가공
temp_Point = []
for i in range(len(x)):
    temp_Point.append(((x[i] - center[0]) / Radius * 144.6 / 94, (y[i] - center[1]) / Radius* 144.6 / 94))


bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 144.6 / 94, enter_editmode=False, location=(0, 0, 0))
obj = bpy.data.objects[0]
obj.name = "face003"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face003']
# select vertex
obj = bpy.context.active_object

for i in range(8, 25):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    diff_x = temp_Point[i - 8][0] - vertex_list[i][0]
    diff_y = -temp_Point[i - 8][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode='OBJECT')

#2번 평면 좌표 가공
flat2_point= copy.deepcopy(NEW_JAMLINE_POINTS)
y_6 = ((ORG_MOUSE_OUTLINE_POINTS[0][1] + ORG_MOUSE_OUTLINE_POINTS[6][1])/2 + flat2_point[6][1])/2
y_7 = ((ORG_MOUSE_OUTLINE_POINTS[7][1] + ORG_MOUSE_OUTLINE_POINTS[11][1])/2 + flat2_point[7][1])/2
y_8 = (ORG_MOUSE_OUTLINE_POINTS[9][1] + flat2_point[8][1])/2
flat2_point[6] = (flat2_point[6][0],y_6)
flat2_point[7] = (flat2_point[7][0],y_7)
flat2_point[8] = (flat2_point[8][0],y_8)
flat2_point[9] = (flat2_point[9][0],y_7)
flat2_point[10] = (flat2_point[10][0],y_6)

temp_list=[]
r1 = (NEW_JAMLINE_POINTS[0][0] + ORG_RIGHT_EYEBROW_POINTS[0][0] )/ 2
r2 = NEW_JAMLINE_POINTS[6][0]
w = r2-r1
h1 = flat2_point[0][1]
h2 = flat2_point[6][1]
w2 = h2-h1
for i in range(6):
#    new_point.append((r1 + (1 - (6-i)*(6-i)/36) * w, NEW_JAMLINE_POINTS[i][1]))
#    new_point.append((r1 + (1 - i*i / 36) * w, NEW_JAMLINE_POINTS[i][1]))
    if i == 5:
        temp_list.append((r1 + w * i ** 2 / 36, (NEW_JAMLINE_POINTS[i][1]+ NEW_JAMLINE_POINTS[i-1][1])/2))
    else:
        temp_list.append((r1 + w * i ** 2 / 36, NEW_JAMLINE_POINTS[i][1]))

for i in range(6):
    flat2_point[i] = (temp_list[i][0], temp_list[i][1])
    flat2_point[-(i+1)] = (2*center[0] - temp_list[i][0], temp_list[i][1])

temp_Point = []
for i in flat2_point:
    temp_Point.append(((i[0] - center[0]) / Radius * 144.6 / 94, (i[1] - center[1]) / Radius* 144.6 / 94))

rr = 1 * (center[0]-flat2_point[0][0])/Radius * 144.6 / 94
print(rr)
bpy.ops.mesh.primitive_circle_add(vertices=32, radius= rr, enter_editmode=False, location=(0, 0, 94/110/2))
obj = bpy.data.objects[0]
obj.name = "face002"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face003']
# select vertex
obj = bpy.context.active_object

for i in range(8, 25):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    diff_x = temp_Point[i - 8][0] - vertex_list[i][0]
    diff_y = -temp_Point[i - 8][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode='OBJECT')

#4번째 (단순 원형)
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 94 / 94, enter_editmode=False, location=(0, 0, -94/110))
obj = bpy.data.objects[0]
obj.name ="face004"

bpy.ops.mesh.primitive_circle_add(vertices=8, radius=0.1, enter_editmode=False, location=(0, 0, -1))
obj = bpy.data.objects[0]
obj.name ="face_back"

#맨앞 평면 제작
face_point= [ORG_RIGHT_EYEBROW_POINTS[4],ORG_RIGHT_EYE_POINTS[0],ORG_MOUSE_OUTLINE_POINTS[0],ORG_MOUSE_OUTLINE_POINTS[10],
             ORG_MOUSE_OUTLINE_POINTS[8],ORG_MOUSE_OUTLINE_POINTS[6],ORG_LEFT_EYE_POINTS[3],ORG_LEFT_EYEBROW_POINTS[1]]

mid_face_x = (face_point[0][0]+face_point[3][0]+face_point[4][0]+face_point[7][0])/4
for i in range(5):
    length = face_point[-(i+1)][0] - face_point[i][0]
    height = (face_point[-(i+1)][1] + face_point[i][1])/2
    face_point[i] = (mid_face_x - length/2, height)
    face_point[-(i+1)] = (mid_face_x + length/2, height)

#print_point(face_point)

temp_Point = []
for i in face_point:
    temp_Point.append(((i[0] - mid_face_x) / Radius * 144.6 / 94, (i[1] - center[1]) / Radius* 144.6 / 94))

bpy.ops.mesh.primitive_circle_add(vertices=8, radius=0.1, enter_editmode=False, location=(0, 0, 1))
obj = bpy.data.objects[0]
obj.name ="face_front"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face_front']
# select vertex
obj = bpy.context.active_object

for i in range(8):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    diff_x = temp_Point[i][0] - vertex_list[i][0]
    diff_y = -temp_Point[i][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode='OBJECT')

#코 평면 제작
nose_point = [ORG_NOSE_POINTS[0],ORG_NOSE_POINTS[4],ORG_NOSE_POINTS[8]]
mid_nose_x = (nose_point[0][1]+nose_point[1][1]+nose_point[2][1])/3

nose_length = nose_point[2][0] - nose_point[1][0]
nose_height = (nose_point[2][1] + nose_point[1][1])/2

nose_point[0] = (mid_nose_x,nose_point[0][1])
nose_point[1] = (mid_nose_x-nose_length/2, nose_height)
nose_point[2] = (mid_nose_x+nose_length/2, nose_height)

temp_Point = []
for i in nose_point:
    temp_Point.append(((i[0] - mid_nose_x) / Radius * 144.6 / 94, (i[1] - center[1]) / Radius* 144.6 / 94))

bpy.ops.mesh.primitive_circle_add(vertices=3, radius=0.1, enter_editmode=False, location=(0, 0, 1))
obj = bpy.data.objects[0]
obj.name ="face_nose"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face_nose']
# select vertex
obj = bpy.context.active_object

for i in range(3):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    diff_x = temp_Point[i][0] - vertex_list[i][0]
    diff_y = -temp_Point[i][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode='OBJECT')

nose_point_end = [ORG_NOSE_POINTS[2],ORG_NOSE_POINTS[2],ORG_NOSE_POINTS[2]]
nose_end_height = ORG_NOSE_POINTS[3][1]+(ORG_NOSE_POINTS[3][1]-ORG_NOSE_POINTS[2][1])/2
nose_point_end[1] = (ORG_NOSE_POINTS[5][0], nose_end_height)
nose_point_end[2] = (ORG_NOSE_POINTS[7][0], nose_end_height)

temp_Point = []
for i in nose_point_end:
    temp_Point.append(((i[0] - nose_point_end[0][0]) / Radius * 144.6 / 94, (i[1] - center[1]) / Radius* 144.6 / 94))

bpy.ops.mesh.primitive_circle_add(vertices=3, radius=0.05, enter_editmode=False, location=(0, 0, 1.2))
obj = bpy.data.objects[0]
obj.name ="face_nose_end"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face_nose_end']
# select vertex
obj = bpy.context.active_object

for i in range(3):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    diff_x = temp_Point[i][0] - vertex_list[i][0]
    diff_y = -temp_Point[i][1] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.export_scene.fbx(filepath="3D/head.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/head.blend")