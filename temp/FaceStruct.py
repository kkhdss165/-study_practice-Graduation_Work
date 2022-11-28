import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

#파일 불러오기
bpy.ops.import_scene.fbx(filepath="3D/circle.fbx")
bpy.ops.object.select_all(action='DESELECT')


bpy.ops.object.select_all(action='SELECT')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB


obj = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')

N = len(obj.data.vertices)

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[i-1].select = True
    obj.data.vertices[i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

# 평면 복사

bpy.ops.object.duplicate_move()
bpy.context.object.location[2] = 0.3
bpy.context.object.scale =(0.9,0.9,0.9)

##복사 스케일 이동
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_pattern(pattern='Circle')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB
bpy.ops.object.duplicate_move()
bpy.context.object.location[2] = 0.6
bpy.context.object.scale =(0.6,0.6,0.6)

##복사 스케일 이동
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_pattern(pattern='Circle')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB
bpy.ops.object.duplicate_move()
bpy.context.object.location[2] = 0.8
bpy.context.object.scale =(0.13,0.13,0.13)

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

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[N+i].select = True
    obj.data.vertices[2*N+i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')


bpy.ops.object.mode_set(mode='EDIT')

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[2*N+i].select = True
    obj.data.vertices[3*N+i].select = True

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode = 'OBJECT')

bpy.ops.object.mode_set(mode='EDIT')

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    obj.data.vertices[3*N+i].select = True
    obj.data.vertices[i].select = True

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

#평면 제작하기2
bpy.ops.object.mode_set(mode='EDIT')

for i in range(N):
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    if i == N-1:
        obj.data.vertices[2 * N + i].select = True
        obj.data.vertices[2 * N + i + 1].select = True
        obj.data.vertices[3 * N + i].select = True
        obj.data.vertices[2 * N].select = True
    else :
        obj.data.vertices[2 * N + i].select = True
        obj.data.vertices[2 * N + i + 1].select = True
        obj.data.vertices[3 * N + i].select = True
        obj.data.vertices[3 * N + i + 1].select = True

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
        obj.data.vertices[3*N + i].select = True    #4N-1
        obj.data.vertices[3*N].select = True        #3N
    else :
        obj.data.vertices[i].select = True
        obj.data.vertices[i + 1].select = True
        obj.data.vertices[3*N + i].select = True
        obj.data.vertices[3*N + i + 1].select = True

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

bpy.ops.export_scene.fbx(filepath="3D/face.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/face.blend")