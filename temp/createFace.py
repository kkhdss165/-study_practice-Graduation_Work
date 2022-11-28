import bpy
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ORG_JAMLINE_POINTS =[(84, 341), (93, 404), (104, 466), (119, 525),
                 (139, 583), (173, 632), (219, 672), (271, 704),
                 (330, 714),
                 (387, 704), (436, 672), (477, 631),(505, 580),
                 (521, 521), (532, 462), (544, 401), (549, 341)]

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

#좌표 가공(좌우 대칭)
R = abs(ORG_JAMLINE_POINTS[0][0] - ORG_JAMLINE_POINTS[16][0])/2
mid_x = (ORG_JAMLINE_POINTS[0][0] + ORG_JAMLINE_POINTS[16][0] + ORG_JAMLINE_POINTS[8][0])/ 3
for i in range(int(len(ORG_JAMLINE_POINTS)/2)):
    len_x = (ORG_JAMLINE_POINTS[-(i+1)][0]-ORG_JAMLINE_POINTS[i][0])/2
    mid_y = (ORG_JAMLINE_POINTS[-(i+1)][1]+ORG_JAMLINE_POINTS[i][1])/2

    NEW_JAMLINE_POINTS[i] = (mid_x-len_x, mid_y)
    NEW_JAMLINE_POINTS[-(i+1)] = (mid_x+len_x, mid_y)

    print(NEW_JAMLINE_POINTS[i][0]+NEW_JAMLINE_POINTS[-(i+1)][0])

NEW_JAMLINE_POINTS[8] = (mid_x,ORG_JAMLINE_POINTS[8][1])
H = NEW_JAMLINE_POINTS[8][1] - NEW_JAMLINE_POINTS[0][1]

center = (NEW_JAMLINE_POINTS[8][0],NEW_JAMLINE_POINTS[0][1])
print(H, R)
print(center)

temp_Point = []
for i in NEW_JAMLINE_POINTS:
    temp_Point.append(((i[0]-center[0])/R, (i[1]-center[1])/R))
print(temp_Point)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1, enter_editmode=False, location=(0, 0, 0.6))
obj = bpy.data.objects[-1]
obj.name ="face002"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects[-1]
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


#두번째
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1, enter_editmode=False, location=(0, 0, 0))
obj = bpy.data.objects[-1]
obj.name ="face003"




bpy.ops.export_scene.fbx(filepath="3D/head.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/head.blend")

