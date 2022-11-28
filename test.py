from create_face_test import createFaceMesh

expansion_ratio, front_point, right_point, back_point = createFaceMesh('./img/screen_20221104235930.png').createMesh()
expansion_ratio, front_point, right_point, back_point = createFaceMesh('./img/hyunbin.png').createMesh()
print(expansion_ratio, front_point, right_point, back_point)