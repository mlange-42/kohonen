..\target\release\kohonen ^
--file ..\example_data\iris.csv ^
--size 16 12 ^
--epochs 250 ^
--layers "sepal_length sepal_width petal_length petal_width" "species" ^
--labels species ^
--categ false true ^
--norm gauss none ^
--weights 1 0.5 ^
--alpha 0.2 0.01 lin ^
--radius 8 0.7 lin ^
--decay 0.2 0.001 exp ^
--neigh gauss ^
--output ..\example_data\_iris ^
--wait