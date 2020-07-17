for i in 1 2 3 4 5 ; do
	python test.py \
		--input testimgs/images/image_00$i.png \
		--mask testimgs/masks/mask_00$i.png \
		--output testimgs/output/result$i \
		--pretrained LBAM_Pretrained_Model/Places10Classes/LBAM_NoBN_Places10Classes.pth
done
