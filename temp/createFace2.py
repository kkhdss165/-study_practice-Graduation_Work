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

new_point=[]
r1 = center[0] - R_94
r2 = NEW_JAMLINE_POINTS[8][0]

w = r2-r1

for i in range(9):
#    new_point.append((r1 + (1 - (6-i)*(6-i)/36) * w, NEW_JAMLINE_POINTS[i][1]))
#    new_point.append((r1 + (1 - i*i / 36) * w, NEW_JAMLINE_POINTS[i][1]))
    new_point.append((r1 + w * i**2 / 64, NEW_JAMLINE_POINTS[i][1]))

for i in range(9):
    NEW_JAMLINE_POINTS[i] = (new_point[i][0], NEW_JAMLINE_POINTS[i][1])
    NEW_JAMLINE_POINTS[-(i+1)] = (2*center[0] - new_point[i][0], NEW_JAMLINE_POINTS[-(i+1)][1])

# 출력
# print(r1,r2,w)
# print(NEW_JAMLINE_POINTS)
# 
# x=[]
# y=[]
# for i in NEW_JAMLINE_POINTS:
#     x.append(i[0])
#     y.append(i[1])
# 
# plt.plot(x,y, marker="X")
# plt.show()



temp_Point = []
for i in NEW_JAMLINE_POINTS:
    temp_Point.append(((i[0]-center[0])/R_94, (i[1]-center[1])/R_94))
print(temp_Point)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

#첫번째
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 94 / 94, enter_editmode=False, location=(0, 0, 110/94))
obj = bpy.data.objects[-1]
obj.name ="face002"
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

obj = bpy.data.objects['face002']
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

#두번째 (x좌표 temp_point 와 기본생성 좌표의 평균으로 설정)

bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 144.6 /94, enter_editmode=False, location=(0, 0, 0))
obj = bpy.data.objects[0]
obj.name ="face003"

obj = bpy.data.objects[-1]
vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(vertex_list)

#모델 좌표 생성
xy_coord=[[vertex_list[12][0],vertex_list[12][1]], [0,-1*Line_Y/Radius * 144.6 /94], [vertex_list[20][0],vertex_list[20][1]]]
xy = np.array(xy_coord)
x =[]
for i in range(12,21):
    print(vertex_list[i][0])
    x.append(vertex_list[i][0])

print(x)
intrp = interp1d(xy[:,0], xy[:,1],  kind='quadratic')
y = intrp(x)
print(y)

#좌표 이동
obj = bpy.data.objects['face003']
#select vertex
obj = bpy.context.active_object

for i in range(12,21):
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    diff_x = x[i-12] - vertex_list[i][0]
    diff_y = y[i-12] - vertex_list[i][1]

    bpy.ops.transform.translate(value=(diff_x, diff_y, 0))
bpy.ops.object.mode_set(mode = 'OBJECT')

#세번째 (단순 원형)
bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1 * 94 / 94, enter_editmode=False, location=(0, 0, -110/94))
obj = bpy.data.objects[0]
obj.name ="face004"

#합치기
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()

##선 잇기
bpy.ops.object.select_all(action='SELECT')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB


obj = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')
N = 32
for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[i].select = True
    obj.data.vertices[2*N+i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

bpy.ops.object.mode_set(mode='EDIT')
for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[N+i].select = True
    obj.data.vertices[2*N+i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

#평면 제작하기
bpy.ops.object.mode_set(mode='EDIT')

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    if i == N-1:
        obj.data.vertices[N + i].select = True
        obj.data.vertices[N + i + 1].select = True
        obj.data.vertices[2 * N + i].select = True
        obj.data.vertices[N].select = True
    else :
        obj.data.vertices[N + i].select = True
        obj.data.vertices[N + i + 1].select = True
        obj.data.vertices[2 * N + i].select = True
        obj.data.vertices[2 * N + i + 1].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

#평면 제작하기3
bpy.ops.object.mode_set(mode='EDIT')

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    if i == N-1:
        obj.data.vertices[i].select = True          #N-1(11)
        obj.data.vertices[0].select = True          #0
        obj.data.vertices[2*N + i].select = True    #4N-1
        obj.data.vertices[2*N].select = True        #3N
    else :
        obj.data.vertices[i].select = True
        obj.data.vertices[i + 1].select = True
        obj.data.vertices[2*N + i].select = True
        obj.data.vertices[2*N + i + 1].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

#마지막 한면 제작하기
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type="VERT")
bpy.ops.mesh.select_all(action = 'DESELECT')
bpy.ops.object.mode_set(mode = 'OBJECT')
for i in range(N):
    obj.data.vertices[i].select = True

bpy.ops.object.mode_set(mode = 'EDIT')
bpy.ops.mesh.edge_face_add()
bpy.ops.object.mode_set(mode = 'OBJECT')

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type="VERT")
bpy.ops.mesh.select_all(action = 'DESELECT')
bpy.ops.object.mode_set(mode = 'OBJECT')
for i in range(N):
    obj.data.vertices[N+i].select = True

bpy.ops.object.mode_set(mode = 'EDIT')
bpy.ops.mesh.edge_face_add()
bpy.ops.object.mode_set(mode = 'OBJECT')

##렌더링
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB

bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 2
bpy.context.object.modifiers["Subdivision"].render_levels = 2
bpy.context.object.modifiers["Subdivision"].quality = 3

bpy.ops.object.shade_smooth()

#오리진 중심으로 오브젝트의 중심으로 이동
bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
bpy.ops.object.select_all(action='DESELECT')

#회전
bpy.context.object.rotation_euler[0] = 1.5708

#색상입히기
#메테리얼 슬롯 제거 추가
bpy.ops.object.material_slot_remove()
bpy.ops.object.material_slot_add()

#메테리얼 설정
new_mat = bpy.data.materials.new("NAME")
new_mat.diffuse_color = (1, 0.730394, 0.502537, 1)
bpy.context.object.active_material = new_mat

bpy.ops.export_scene.fbx(filepath="3D/head.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/head.blend")

