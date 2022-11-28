import cv2
import mediapipe as mp
import bpy

OUTLINE_POINTS_1 = [10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338]    #36개
OUTLINE_POINTS_2 = [151,108,69,104,68,71,139,34,227,137,177,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337]  #36개
OUTLINE_POINTS_3 = [9,107,66,105,63,70,156,143,116,123,147,213,192,214,210,211,32,208,199,428,262,431,430,434,416,433,376,352,345,372,383,300,293,334,296,336]    #36개
OUTLINE_POINTS_4 = [8,55,65,52,53,46,124,35,111,117,50,187,207,216,212,202,204,194,201,200,421,418,424,422,432,436,427,411,280,346,340,265,353,276,283,282,295,285] #38개

#눈
RIGHT_EYE_POINT_1 = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   #16개
RIGHT_EYE_POINT_2 = [130,25,110,24,23,22,26,112,243,190,56,28,27,29,30,247]   #16개
RIGHT_EYE_POINT_3 = [226,31,228,229,230,231,232,233,244,189,221,222,223,224,225,113]   #16개

LEFT_EYE_POINT_1 = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]   #16개
LEFT_EYE_POINT_2 = [463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414]   #16개
LEFT_EYE_POINT_3 = [464,453,452,451,450,449,448,261,446,342,445,444,443,442,441,413]   #16개

#입술
MOUSE_POINT_1 = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]  #20개
MOUSE_POINT_2 = [62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]  #20개
MOUSE_POINT_3 = [76,77,90,180,85,16,315,404,320,307,306,408,304,303,302,11,72,73,74,184]  #20개
MOUSE_POINT_4 = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]  #20개
MOUSE_POINT_5 = [57,43,106,182,83,18,313,406,335,273,287,410,322,391,393,164,167,165,92,186] #20개

def delete_all_object():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
def export_bpy(filename):
    bpy.ops.export_scene.fbx(filepath="3D/"+filename+".fbx")
    bpy.ops.wm.save_as_mainfile(filepath="3D/"+filename+".blend")

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

def create_landmarks_list(landmarks_index):
    list =[]
    for facial_landmarks in result.multi_face_landmarks:

        for i in landmarks_index:
            pt1 = facial_landmarks.landmark[i]
            list.append((pt1.x, pt1.y, pt1.z))

    return list


#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

image = cv2.imread('../img/hyunbin.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = face_mesh.process(rgb_image)
height, width, _ = image.shape


if __name__ == '__main__':
    delete_all_object()

    temp_Point = create_landmarks_list(OUTLINE_POINTS_1)
    create_plane(temp_Point, "OUTLINE1")
    temp_Point = create_landmarks_list(OUTLINE_POINTS_2)
    create_plane(temp_Point, "OUTLINE2")
    temp_Point = create_landmarks_list(OUTLINE_POINTS_3)
    create_plane(temp_Point, "OUTLINE3")
    temp_Point = create_landmarks_list(OUTLINE_POINTS_4)
    create_plane(temp_Point, "OUTLINE4")

    temp_Point = create_landmarks_list(MOUSE_POINT_1)
    create_plane(temp_Point, "mouse1")
    temp_Point = create_landmarks_list(MOUSE_POINT_2)
    create_plane(temp_Point, "mouse2")
    temp_Point = create_landmarks_list(MOUSE_POINT_3)
    create_plane(temp_Point, "mouse3")
    temp_Point = create_landmarks_list(MOUSE_POINT_4)
    create_plane(temp_Point, "mouse4")
    temp_Point = create_landmarks_list(MOUSE_POINT_5)
    create_plane(temp_Point, "mouse5")

    temp_Point = create_landmarks_list(RIGHT_EYE_POINT_1)
    create_plane(temp_Point, "righteye1")
    temp_Point = create_landmarks_list(RIGHT_EYE_POINT_2)
    create_plane(temp_Point, "righteye2")
    temp_Point = create_landmarks_list(RIGHT_EYE_POINT_3)
    create_plane(temp_Point, "righteye3")

    temp_Point = create_landmarks_list(LEFT_EYE_POINT_1)
    create_plane(temp_Point, "lefteye1")
    temp_Point = create_landmarks_list(LEFT_EYE_POINT_2)
    create_plane(temp_Point, "lefteye2")
    temp_Point = create_landmarks_list(LEFT_EYE_POINT_3)
    create_plane(temp_Point, "lefteye3")


    export_bpy("head")