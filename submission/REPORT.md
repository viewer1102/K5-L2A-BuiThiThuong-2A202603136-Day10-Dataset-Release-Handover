# Day 10 — Báo cáo cá nhân

## Tự đo — seed 11

- Số lỗi nộp: 17
- F1 tổng ≈ 1.0 · lỗi luật ≈ 1.0 · lỗi gán nhãn ≈ 0.9
- Truy vết F1 ≈ tier_a 1.0 · tier_b 1.0 · notify 1.0

Chỉ in làm tròn: dùng để biết validator tốt lên hay kém đi, không để dò từng ảnh.

## Hàng đợi review
- Ảnh cần xem (seed luyện tập): 5 ảnh đối chiếu nhãn và 2 ảnh kiểm tra wrong_hash.
- Lô sạch 50 ảnh (seed 5, `--no-faults`): validator ghi nhận 0 lỗi luật và sinh 1 ứng viên đối chiếu nhãn.
- Ngưỡng khởi tạo theo IoU, phần pixel mask nằm trong box và diện tích segment. Các ca lệch tự nhiên được ghi rejected trong review_log.csv. Nếu đưa cả các ca đó vào errors.json, false positive tăng, precision giảm và kéo F1 xuống.

## Quyết định và lineage
QC v1: 17/250 = 6.80%, kết luận Hold. R-021 đã dùng v1, nên cần đánh giá lại hoặc train lại bằng bản phù hợp.
Lô `lab250_60b9be68a82e_v2` ghi v1 là lô cha. V2 giữ 243 ảnh với
1.683 nhãn và chuyển 7 ảnh sang `rework_queue.csv`. Manifest bao phủ
486 file ảnh và nhãn. Bộ kiểm bàn giao đạt 17 PASS, 0 FAIL.

## Ví dụ review thủ công

- `000000007784`: Vật trong ảnh là diều. YOLO gán `bird`, còn panoptic
  gán `kite`. Xác nhận `class_swap`.
- `000000006213`: Vùng khoanh là bồn rửa mặt. YOLO gán `sink`, còn
  panoptic gán `frisbee`. Xác nhận `category_swap`.
- `000000001818`: Mask tách riêng hai con ngựa vằn. Box YOLO và segment
  đang xét thuộc hai con khác nhau. Bác bỏ `bbox_drift` vì script ghép nhầm.

## SHA-256
Hash giống nhau cho thấy file khớp về byte với bản tham chiếu. Muốn biết box và lớp đúng cần xem ảnh và áp dụng guideline.

## Truy vết
Tầng A là ảnh trong cảnh bị báo, thuộc job dùng guideline cũ và có box của lớp đổi quy tắc. Tầng B là các ảnh cùng điều kiện ở cảnh khác. Dùng phiên bản theo job để tránh metadata ảnh bị sai.
