# Bank Card Number Recognition with CRNN

Use CTPN method to detect bank card number and CRNN method to finish the recognition process.

ctpn_paper:https://arxiv.org/abs/1609.03605

ctpn_implementation:https://github.com/eragonruan/text-detection-ctpn

follow the instruction to setup, because those codes are written in Cython according to the author:

```
cd ctpn/utils/bbox
chmod +x make.sh
./make.sh
```

- [x] download the  [pre-trained model](https://drive.google.com/open?id=11QexbXFjDmz-ww5gULSlE2IUwiWxQCpn) to bank-card-number-detection-CRNN/ctpn/checkpoints_mlt

crnn_implementation:[https://github.com/DevilExileSu/BankCardOCR](https://github.com/DevilExileSu/BankCardOCR)

- [ ] download the [pre-trained model]() to bank-card-number-detection-CRNN/crnn/model (not released at present)

**Demo**

![](/res_recognition/(12).jpg)

![](/res_recognition/(14).jpg)

![](/res_recognition/(17).jpg)