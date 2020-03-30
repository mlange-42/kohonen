..\target\release\kohonen.exe ^
--file ..\example_data\us-congress-votes.csv ^
--size 16 12 ^
--epochs 1000 ^
--layers "handicapped-infants water-project-cost-sharing adoption-of-the-budget-resolution physician-fee-freeze el-salvador-aid religious-groups-in-schools anti-satellite-test-ban aid-to-nicaraguan-contras mx-missile immigration synfuels-corporation-cutback education-spending superfund-right-to-sue crime duty-free-exports export-administration-act-south-africa" "party" ^
--labels label ^
--label-length 10 ^
--categ false true ^
--norm none none ^
--weights 1 0.5 ^
--alpha 0.2 0.01 lin ^
--radius 6 0.8 lin ^
--decay 0.2 0.001 exp ^
--neigh gauss ^
--no-data - ^
--fps 1 ^
--output ..\example_data\_us-congress-votes
