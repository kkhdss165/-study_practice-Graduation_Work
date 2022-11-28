import cv2
import mediapipe as mp
import bpy


#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

image = cv2.imread('../img/hyunbin.png')
#image = cv2.imread('../img/jungmin.png')
#image = cv2.imread('../img/sangsoo.png')
#image = cv2.imread('../img/sukhun.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = face_mesh.process(rgb_image)
height, width, _ = image.shape

SCALE_VALUE = 2.0

#모든 오브젝트 삭제
def delete_all_object():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
def import_bpy(filename):
    bpy.ops.import_scene.fbx(filepath="3D/"+filename+".fbx")
def export_bpy(filename):
    bpy.ops.export_scene.fbx(filepath="3D/"+filename+".fbx")
    bpy.ops.wm.save_as_mainfile(filepath="3D/"+filename+".blend")
#얼굴 랜드마크 생성
def set_landmarks_list():
    list =[]
    for facial_landmarks in result.multi_face_landmarks:

        for i in range(468):
            pt1 = facial_landmarks.landmark[i]
            list.append((pt1.x, pt1.y, pt1.z))

    return list
#리스트의 맞는 얼굴랜드마크 설정
def set_landmarks_by_list(landmarks_index):
    list =[]
    for facial_landmarks in result.multi_face_landmarks:

        for i in landmarks_index:
            pt1 = facial_landmarks.landmark[i]
            list.append((pt1.x*SCALE_VALUE, pt1.y*SCALE_VALUE, pt1.z*SCALE_VALUE))

    return list
#list의 좌표에 매칭되는 평면 생성
def create_plane(list, object_name):
    bpy.ops.mesh.primitive_circle_add(vertices=len(list), radius=0.1, enter_editmode=False, location=(0, 0, 1))
    obj = bpy.data.objects[0]
    obj.name = object_name
    vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

    obj = bpy.data.objects[object_name]
    # select vertex
    obj = bpy.context.active_object

    for i in range(len(list)):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        obj.data.vertices[i].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        diff_x = list[i][0] - vertex_list[i][0]
        diff_y = list[i][1] - vertex_list[i][1]
        diff_z = list[i][2] - vertex_list[i][2]

        bpy.ops.transform.translate(value=(diff_x, diff_y, diff_z))
    bpy.ops.object.mode_set(mode='OBJECT')
#좌표값들을 이용하여 평면 제작
def create_Face(list, object_name):
    obj = bpy.data.objects[object_name]
    # select vertex
    obj = bpy.context.active_object

    for i in list:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        obj.data.vertices[i[0]].select = True
        obj.data.vertices[i[1]].select = True
        obj.data.vertices[i[2]].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.edge_face_add()

    bpy.ops.object.mode_set(mode='OBJECT')
#리스트에 맞게 점들을 이동
def move_mesh_vertex_by_list(list, object_name):

    bpy.ops.object.select_all(action='SELECT')

    bpy.ops.object.select_pattern(pattern='object_name')
    OB = bpy.context.selected_objects[0]
    OB.select_set(state=True)
    bpy.context.view_layer.objects.active = OB

    obj = bpy.data.objects[0]
    obj.name = object_name
    vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

    obj = bpy.data.objects[object_name]
    # select vertex
    obj = bpy.context.active_object

    for i in range(len(list)):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        obj.data.vertices[i].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        diff_x = list[i][0] - vertex_list[i][0]
        diff_y = list[i][1] - vertex_list[i][1]
        diff_z = list[i][2] - vertex_list[i][2]

        bpy.ops.transform.translate(value=(diff_x, diff_y, diff_z))
    bpy.ops.object.mode_set(mode='OBJECT')

def move_origin_center():
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    bpy.context.object.location = (0, 0, 0)
    bpy.context.object.rotation_euler[0] = 1.5708
    bpy.context.object.rotation_euler[1] = 1.5708 * 2
    bpy.context.object.rotation_euler[2] = 1.5708 * 2





if __name__ == '__main__':
    delete_all_object()
    import_bpy("facemesh")
    landmark_List = set_landmarks_list()
    bpy.ops.object.select_all(action='SELECT')
    move_mesh_vertex_by_list(landmark_List, "facemesh")
    move_origin_center()

    export_bpy('new')