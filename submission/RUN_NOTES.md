# Ghi chú lượt chạy T4

- Đã chạy NB0–NB4 trên notebook Colab của người học; bonus chưa chạy.
- Notebook có output: `colab/Lab22_DPO_T4_executed.ipynb`. Đã kiểm tra hợp lệ bằng nbformat.
- `submission/VERIFY_COLAB.txt` ghi kết quả verify tại `/content/lab22`: mã 1, chỉ còn họ tên và khóa học để người học điền.
- Bốn kiểm tra smoke của repo đều qua. Hash 15 file bằng chứng tải về khớp Colab. Các score reward đều hữu hạn.
- Hai adapter có trọng số thật tại `adapters/sft-mini/adapter_model.safetensors` và `adapters/dpo/adapter_model.safetensors`, mỗi file 132187888 byte.
- Bản sao lưu local: `models/lab22-t4-adapters.zip`, 288806996 byte; SHA256 `00a796e8124b944c92222866b0e00abe48466a6321bfeb0633f4e366e993d7cb`. Đây là archive adapter và tokenizer, không chứa merged SFT.
- Mô hình SFT đã merge (~8 GB) còn trên Colab tại `/content/lab22/models/sft-merged`; máy local chỉ có config của merged model. Có thể khôi phục merged SFT từ mô hình gốc và adapter SFT đã sao lưu.
- Adapter DPO giữ đường dẫn reference thật của lượt huấn luyện là `/content/lab22/models/sft-merged`. Không đổi config sang đường dẫn Windows để giả lập verify thành công. `verify` trên máy local sẽ báo reference khác vị trí; báo cáo Colab là bằng chứng kiểm tra lượt chạy gốc.
- Trọng số/archive được gitignore theo quy định repo; các cấu hình, metrics, parquet, JSON/JSONL, ảnh, notebook và phản tư là bằng chứng nhỏ để nộp.
- Sau khi điền thông tin cá nhân, đồng bộ `submission/REFLECTION.md` sang runtime Colab rồi chạy lại `!python scripts/verify.py`. Chưa nộp LMS.
