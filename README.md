# EGCO461: Thai Sentiment Classification

โครงงานจำแนกข้อความภาษาไทยเป็น 4 คลาส ได้แก่ `pos` (เชิงบวก), `neu` (เป็นกลาง), `neg` (เชิงลบ) และ `q` (คำถามเกี่ยวกับสินค้าหรือบริการ) โดยใช้ **macro-F1** เป็นตัวชี้วัดหลัก

## โครงสร้าง repository

```text
code/
  data/                         ข้อมูล train, valid, sample, test และตัวอย่างไฟล์ส่ง
  baseline_naive_bayes.ipynb    CountVectorizer + Multinomial Naive Bayes
  person1.ipynb                 Word TF-IDF + Logistic Regression
  person2.ipynb                 Character TF-IDF + Linear SVM
  person3.ipynb                 Fine-tune WangchanBERTa
predictions/                    ไฟล์ผลทำนายที่เตรียมไว้
```

ไฟล์ข้อมูลประกอบด้วย `train.csv` 24,020 แถวสำหรับฝึก, `valid.csv` 2,669 แถวสำหรับเลือกโมเดล, `sample_labeled.csv` 100 แถวสำหรับตรวจแนวโน้มกับข้อมูลรูปแบบใกล้ test, `test.csv` 1,577 แถวที่ไม่มีเฉลย และ `sample_submission.csv` สำหรับตรวจรูปแบบผลทำนาย โดย sample มีคลาส `q` เพียง 5 ตัวอย่าง จึงควรระวังการตีความคะแนน

## เตรียมสภาพแวดล้อม

ติดตั้ง Python และ Jupyter Notebook หรือ JupyterLab แล้วติดตั้งแพ็กเกจสำหรับ baseline และแนวทางที่ 1–2:

```bash
python -m pip install jupyter pandas numpy scipy scikit-learn pythainlp
```

เปิด terminal ที่โฟลเดอร์ repository แล้วใช้ `jupyter lab` หรือ `jupyter notebook` เปิด notebook และรันเซลล์ตามลำดับ การติดตั้งแพ็กเกจอาจต้องใช้อินเทอร์เน็ต ควรบันทึกเวอร์ชันที่ใช้เมื่อต้องการทำซ้ำผล

แนวทางที่ 3 ใช้ PyTorch และ Transformers พร้อมโมเดล pretrained ซึ่งต้องดาวน์โหลด แนะนำ Google Colab ที่เปิด GPU หรือเครื่องที่ติดตั้ง PyTorch รองรับ GPU เซลล์แรกของ `person3.ipynb` ติดตั้ง `transformers==4.46.3`, `accelerate`, `sentencepiece`, `pythainlp` และ `scikit-learn`; ตรวจว่า pandas, numpy และ PyTorch พร้อมใช้งานด้วย

## ลำดับการใช้งาน

1. **Baseline:** เปิด `code/baseline_naive_bayes.ipynb` โดยให้ working directory เป็น `code/` เพราะ notebook อ่าน `data/*.csv` แบบ relative รันทุกเซลล์เพื่อฝึก baseline และดูคะแนน เซลล์สุดท้ายเขียน `G00_run1_nb_baseline.csv` ใน working directory; เปลี่ยนชื่อหรือย้ายไฟล์ตามที่ทีมตกลงก่อนใช้ส่ง
2. **แนวทางที่ 1:** เปิด `code/person1.ipynb` จาก root หรือ `code/` แล้วรันตามลำดับ Notebook เปรียบเทียบ Logistic Regression แบบปกติและ balanced บน valid และ sample ก่อนฝึกโมเดลสุดท้ายด้วย train+valid ผลการทดลองบันทึกใน `experiments/person1_tfidf_lr_balanced/` และสร้าง `predictions/dingdong_run1_tfidf_lr_balanced.csv`
3. **แนวทางที่ 2:** เปิด `code/person2.ipynb` จาก root หรือ `code/` แล้วรันตามลำดับ Notebook เปรียบเทียบ character n-gram, ค่า C และ class weight โดยเลือกจาก valid; ประเมิน sample ควบคู่กัน เซลล์สุดท้ายใช้ `GROUP_ID` สร้าง `predictions/dingdong_run2_char_tfidf_svm.csv` เมื่อกำหนดค่าเป็น `dingdong` โมเดลนี้ฝึกจาก train
4. **แนวทางที่ 3:** เปิด `code/person3.ipynb` และรันตามลำดับ ใช้ train ฝึกและ valid เลือก checkpoint และค่า bias ผลการทดลองและ checkpoints อยู่ใน `experiments/person3_wangchanberta/` เซลล์สุดท้ายสร้างไฟล์ใน `predictions/` โดยใช้ `GROUP_ID` และ `RUN_ID` ที่กำหนดใน notebook ตรวจชื่อไฟล์ที่พิมพ์หลังรัน เนื่องจากชื่อที่สร้างอาจต่างจากไฟล์เดิมใน repository

Notebook แนวทางที่ 1 และ 3 ค้นหาโฟลเดอร์ `code/data` จาก working directory และ parent ได้ หากใช้ Colab ให้วางโครงสร้าง `code/data` ให้ครบและเปิดจากโฟลเดอร์โครงงาน หรือเก็บโฟลเดอร์ชื่อ `egco461-thai-sentiment` ไว้ใน Google Drive แล้วทำตามขั้นตอน mount ใน notebook

การรันเซลล์สร้างผลทำนายอาจเขียนทับไฟล์ชื่อเดียวกัน ควรตรวจ `GROUP_ID`, `RUN_ID` และชื่อปลายทางก่อนรัน `experiments/` และไฟล์โมเดลขนาดใหญ่เป็นผลลัพธ์จากการรันและอาจไม่มีอยู่ใน checkout เริ่มต้น

## ตรวจไฟล์ก่อนส่ง

ใน `predictions/` ปัจจุบันมีไฟล์สำหรับทั้ง 3 แนวทาง ตรวจแต่ละไฟล์เทียบ `code/data/sample_submission.csv`: ต้องเป็น CSV UTF-8 มีคอลัมน์ `id,category`, มี 1,577 แถว, id ครบและไม่ซ้ำ และ category เป็น `pos`, `neu`, `neg` หรือ `q` เท่านั้น ตรวจชื่อกลุ่มตามที่อาจารย์กำหนดและตรวจว่า notebook ที่ใช้รันสร้างไฟล์นั้นได้จริง

ตามโจทย์ ให้ส่ง ZIP หนึ่งไฟล์ภายใน **24 พฤศจิกายน 2569 เวลา 23:59 น.** ประกอบด้วย `predictions/` ไม่เกิน 5 ไฟล์, `report.pdf` และ `code/` พร้อมคำอธิบายวิธีรัน รายงานต้องมีผล baseline และอย่างน้อย 3 แนวทางบน valid และ sample, F1 รายคลาส, ablation, เหตุผลเลือกไฟล์, ตัวอย่าง error analysis อย่างน้อย 10 ข้อความ และระบุการใช้ AI รายงานหลักยาวไม่เกิน 6 หน้าโดยไม่นับภาคผนวก ตรวจให้แน่ใจว่าได้เพิ่ม `report.pdf` ก่อนจัด ZIP

ห้ามใช้ `test.csv` เพื่อฝึกหรือเลือก hyperparameters, ติด label test ด้วยมือ หรือส่งข้อความ test ให้ LLM/บริการภายนอกทำนาย ไฟล์ที่ส่งต้องมาจากโมเดลที่ทีมฝึกเอง ใช้ valid เลือกโมเดลและใช้ sample เป็นข้อมูลประกอบ เมื่อเลือกเสร็จแล้วโจทย์อนุญาตให้รวม train กับ valid เพื่อฝึกโมเดลสุดท้าย

แหล่งข้อมูล train/valid: Wisesight Sentiment Corpus (CC0), Suriyawongkul et al., DOI: 10.5281/zenodo.3457446
