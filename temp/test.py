import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


bpy.ops.import_scene.fbx(filepath="3D/apple.fbx")
bpy.ops.object.select_all(action='DESELECT')
#bpy.ops.object.select_pattern(pattern='apple')

#for obj in bpy.data.objects:
#    print(obj.name)

#print(list(bpy.data.objects))
obj = bpy.data.objects['apple']
print(obj.name)

list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print(list)

# obj = bpy.context.active_object
#list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
#print(len(list))

#bpy.ops.wm.save_as_mainfile(filepath="3D/result2.blend", compress=False)
#bpy.ops.export_scene.fbx(filepath="3D/result2.fbx")
