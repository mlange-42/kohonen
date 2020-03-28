..\target\release\kohonen.exe ^
--file ..\example_data\countries.csv ^
--size 20 16 ^
--episodes 10000 ^
--layers "child_mort_2010 birth_p_1000 GNI LifeExpectancy PopGrowth PopUrbanized PopGrowthUrb AdultLiteracy PrimSchool Income_low_40 Income_high_20" "continent" ^
--categ 0 1 ^
--norm gauss none ^
--weights 1 1 ^
--alpha 0.2 0.01 lin ^
--radius 10 0.8 lin ^
--decay 0.2 0.001 exp ^
--neigh gauss ^
--no-data - ^
--fps 1