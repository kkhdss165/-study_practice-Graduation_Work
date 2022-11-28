import bpy

#디폴트 생성 제거
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

#파일 불러오기
bpy.ops.import_scene.fbx(filepath="3D/Body.fbx")
bpy.ops.import_scene.fbx(filepath="3D/Head.fbx")
bpy.ops.object.select_all(action='DESELECT')


#오리진 중심으로 오브젝트의 중심으로 이동
bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
bpy.ops.object.select_all(action='DESELECT')

bpy.data.objects['Head'].location=(0,0,0)
bpy.data.objects['Body'].location=(0,0,0)

#몸체
obj = bpy.data.objects['Body']
list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(list)
min_body_z = list[0][2]
max_body_z = list[0][2]
for i in list:
    if i[2] <= min_body_z:
        min_body_z = i[2]
    if i[2] >= max_body_z:
        max_body_z = i[2]

len_z_body = max_body_z-min_body_z
print(len_z_body)
bpy.data.objects['Body'].location=(0,0,0)

#얼굴
obj = bpy.data.objects['Head']
list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(list)
min_head_z = list[0][2]
max_head_z = list[0][2]
for i in list:
    if i[2] <= min_head_z:
        min_head_z = i[2]
    if i[2] >= max_head_z:
        max_head_z = i[2]

len_z_head = max_head_z - min_head_z
print(len_z_head)
bpy.data.objects['Head'].location=(0,0,(len_z_body/2 + len_z_head/2))


bpy.ops.export_scene.fbx(filepath="3D/Player.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/result2.blend")
