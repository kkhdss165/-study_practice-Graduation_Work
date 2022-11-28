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

#오브젝트 선택하기(노란색)
bpy.ops.object.select_pattern(pattern='Body')
OB = bpy.context.selected_objects[0]
OB.select_set(state=True)
bpy.context.view_layer.objects.active = OB

#메테리얼 슬롯 제거 추가
bpy.ops.object.material_slot_remove()

bpy.ops.object.material_slot_add()

#메테리얼 설정
new_mat = bpy.data.materials.new("NAME")
new_mat.diffuse_color = (1, 0, 0, 1)
bpy.context.object.active_material = new_mat

bpy.data.objects['Head'].location=(0,0,0)
bpy.data.objects['Body'].location=(0,0,0)

bpy.ops.export_scene.fbx(filepath="3D/Player.fbx")
bpy.ops.wm.save_as_mainfile(filepath="3D/result2.blend")
