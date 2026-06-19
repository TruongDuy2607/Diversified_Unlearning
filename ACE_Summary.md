# ACE: Anti-Editing Concept Erasure in Text-to-Image Models

> **Paper:** ACE: Anti-Editing Concept Erasure in Text-to-Image Models  
> **Authors:** Zihao Wang, Yuxiang Wei, Fan Li, Renjing Pei, Hang Xu, Wangmeng Zuo  
> **Affiliation:** Harbin Institute of Technology, Huawei Noah's Ark Lab, Pazhou Lab  
> **ArXiv:** 2501.01633v1 (3 Jan 2025)

---

## 1. Bối cảnh và Vấn đề

Các mô hình text-to-image (T2I) như Stable Diffusion có thể bị lợi dụng để sinh nội dung không phù hợp (ảnh vi phạm bản quyền, nội dung khiêu dâm, v.v.). Để ngăn chặn điều này, các phương pháp **concept erasure** (xóa khái niệm) được đề xuất nhằm "unlearn" khái niệm mục tiêu khỏi model.

### Hạn chế của phương pháp hiện tại

Phương pháp tiêu biểu như **ESD (Erasing Concepts from Diffusion Models)** chỉ xóa khái niệm khỏi **conditional noise prediction**:

$$\tilde{\epsilon}_c = \epsilon_{\theta^*}(z_t, t) - \eta_c \left(\epsilon_{\theta^*}(z_t, c, t) - \epsilon_{\theta^*}(z_t, t)\right) \tag{1}$$

Cơ chế bảo vệ chỉ được kích hoạt khi text prompt **chứa tên khái niệm** (ví dụ: "Pikachu", "nudity"). Do đó:

- Người dùng vẫn có thể dùng **image editing** với prompt trung tính ("Add sunglasses") trên ảnh chứa khái niệm đã xóa → guard không bị kích hoạt → ảnh vẫn được tạo ra.
- Điểm yếu: model dựa vào **text trigger** mà bỏ qua **visual content** của ảnh đầu vào.

### Mục tiêu của ACE

Ngăn chặn việc tạo ra nội dung không mong muốn qua **cả hai con đường**:
1. **Generation** — sinh ảnh từ text prompt.
2. **Editing** — chỉnh sửa ảnh hiện có bằng editing method (LEDITS++, MasaCtrl, SD-Inpainting, v.v.).

---

## 2. Nền tảng lý thuyết

### 2.1 Stable Diffusion & Latent Diffusion

Model T2I được huấn luyện tối thiểu hóa:

$$\mathcal{L}_{LDM} = \mathbb{E}_{z_t, t, c, \epsilon \sim \mathcal{N}(0,I)} \left\| \epsilon - \epsilon_\theta(z_t, c, t) \right\|_2^2 \tag{2}$$

Trong đó:
- $z_t$: latent image được thêm noise ở timestep $t$
- $c$: text embedding (từ text encoder)
- $\epsilon_\theta$: U-Net dự đoán noise

### 2.2 Classifier-Free Guidance (CFG)

Trong inference, CFG kết hợp conditional và unconditional noise prediction để cải thiện chất lượng ảnh:

$$\tilde{\epsilon} = \epsilon_\theta(z_t, t) + \omega \left(\epsilon_\theta(z_t, c, t) - \epsilon_\theta(z_t, t)\right) \tag{3}$$

Trong đó:
- $\epsilon_\theta(z_t, t)$: **unconditional noise prediction** (không có text)
- $\epsilon_\theta(z_t, c, t)$: **conditional noise prediction** (có text)
- $\omega$: guidance scale (thường > 1)

Mối quan hệ với classifier gradient:

$$\nabla_{z_t} \log p(c | z_t) = -\frac{1}{\sigma_t} \left(\epsilon_\theta(z_t, c, t) - \epsilon_\theta(z_t, t)\right) \tag{4}$$

---

## 3. Phương pháp ACE

ACE đề xuất ba thành phần chính:

### 3.1 Unconditional Erasure Guidance (UEG)

**Ý tưởng:** Nếu chỉ xóa khái niệm từ conditional noise, editing vẫn có thể bypass vì text prompt không chứa tên khái niệm. Giải pháp là xóa khái niệm từ **cả unconditional noise prediction** — khi đó CFG sẽ tự động đẩy kết quả ra xa khái niệm mục tiêu **bất kể text input là gì**.

**Định nghĩa UEG:**

$$\tilde{\epsilon}_u = \epsilon_{\theta^*}(z_t, t) + \eta_u \left(\epsilon_{\theta^*}(z_t, c, t) - \epsilon_{\theta^*}(z_t, t)\right) \tag{5}$$

Hướng $\epsilon_{\theta^*}(z_t, c, t) - \epsilon_{\theta^*}(z_t, t)$ chính là hướng của target concept. UEG kéo unconditional noise **theo hướng target concept**, khiến CFG trong inference sẽ tự động **đẩy ra khỏi** target concept.

**Loss function để align unconditional noise với UEG:**

$$\mathcal{L}_{Unc} = \mathbb{E}_{z_t, t, c} \left\| \epsilon_\theta(z_t, t) - \tilde{\epsilon}_u \right\|_2^2 \tag{6}$$

**Phân tích lý thuyết:** Sau khi training với UEG, CFG prediction trở thành:

$$\tilde{\epsilon} \approx \epsilon_{\theta^*}(z_t, t) - \frac{1}{\sigma_t}\left[\eta_u(1-\omega)\nabla_{z_t}\log p(c|z_t) + \omega \nabla_{z_t}\log p(c_{input}|z_t)\right] \tag{7}$$

Cả hai gradient đều có dấu **âm** trước $\nabla \log p(c|\cdot)$ (với $\omega > 1$), nghĩa là quá trình denoising **giảm xác suất** xuất hiện target concept $c$ trong ảnh, bất kể $c_{input}$ là gì.

### 3.2 Prior-Guided Unconditional Erasure Guidance (PG-UEG)

**Vấn đề của UEG thuần túy:**
- UEG kéo unconditional noise gần với target concept → vô tình làm tăng xác suất xuất hiện target concept khi sinh ảnh với unconditional noise (không có text).
- Gây **concept erosion**: ảnh hưởng sang các khái niệm không liên quan.

**Giải pháp:** Trừ đi noise guidance của một **prior concept ngẫu nhiên** $c_p$ từ UEG:

$$\tilde{\epsilon}^p_u = \epsilon_{\theta^*}(z_t, t) + \eta_u \left(\epsilon_{\theta^*}(z_t, c, t) - \epsilon_{\theta^*}(z_t, t)\right) - \eta_p \gamma_p \left(\epsilon_{\theta^*}(z_t, c_p, t) - \epsilon_{\theta^*}(z_t, t)\right) \tag{8}$$

Trong đó:
- $c_p$: prior concept được lấy ngẫu nhiên từ tập $C_p$ (các khái niệm cần bảo tồn)
- $\eta_p$: correction guidance scale
- $\gamma_p$: hệ số kiểm soát, tính bằng CLIP score tương đối:

$$\gamma_p = \frac{\text{CLIP}(x, c_p)}{\text{CLIP}(x, c)}$$

$\gamma_p$ đo mức độ liên quan của prior concept với target concept — prior càng gần target thì correction càng mạnh.

**Loss function PG-UEG:**

$$\mathcal{L}_{PUnc} = \mathbb{E}_{z_t, t, c, c_p \in C_p} \left\| \epsilon_\theta(z_t, t) - \tilde{\epsilon}^p_u \right\|_2^2 \tag{9}$$

**Tác động:** Unconditional noise được "đẩy khỏi" target concept nhưng đồng thời không bị kéo quá gần với bất kỳ prior concept cụ thể nào, giữ khoảng cách cân bằng trong không gian ngữ nghĩa.

### 3.3 Prior Concept Preservation (LCons)

**Vấn đề:** Fine-tuning để xóa target concept có thể vô tình làm suy giảm khả năng sinh các khái niệm liên quan (prior concept erosion).

**Giải pháp:** Regularization loss giữ conditional noise prediction của prior concept không thay đổi so với model gốc:

$$\mathcal{L}_{Cons} = \mathbb{E}_{z_t, t, c_p \in C_p} \left\| \epsilon_\theta(z_t, c_p, t) - \epsilon_{\theta^*}(z_t, c_p, t) \right\|_2^2 \tag{10}$$

**Xây dựng tập prior $C_p$:** Sử dụng LLM (GPT-4) để xác định các khái niệm liên quan ngữ nghĩa với target concept, sau đó dùng làm prior. Việc có nhiều prior tốt hơn nhưng cũng tốn kém hơn về tính toán.

### 3.4 Objective Function Tổng Hợp

$$\mathcal{L}_{ACE} = \lambda_{PUnc} \mathcal{L}_{PUnc} + \lambda_{Cons} \mathcal{L}_{Cons} + \lambda_{ESD} \mathcal{L}_{ESD} \tag{11}$$

Trong đó $\mathcal{L}_{ESD}$ là loss gốc của ESD để xóa khái niệm khỏi conditional noise prediction (giữ nguyên từ ESD):

$$\mathcal{L}_{ESD} = \mathbb{E}_{z_t, t, c} \left\| \epsilon_\theta(z_t, c, t) - \tilde{\epsilon}_c \right\|_2^2 \tag{12}$$

---

## 4. Chi tiết Triển khai

| Cấu hình | Giá trị |
|---|---|
| Base model | Stable Diffusion v1.4 |
| Fine-tuning | LoRA (rank = 4) |
| Learning rate | 0.001 |
| Guidance scale $\eta_u$, $\eta_c$ | 3 |
| CFG scale cho training | 3 |
| DDIM sampling steps | 30 |
| Batch size | 1 |
| Prior sampling batch size | 2 |

**Hyperparameters theo task:**

| Hyperparameter | IP Character | Explicit (Nudity) | Artist Style |
|---|---|---|---|
| Training steps | 1500 | 2000 | 750 |
| $\eta_p$ | 3 | 1 | 1.5 |
| $\lambda_{PUnc}$ | 0.19 | 0.198 | 0.05 |
| $\lambda_{Cons}$ | 0.8 | 0.8 | 0.9 |
| $\lambda_{ESD}$ | 0.01 | 0.002 | 0.05 |

---

## 5. Sơ đồ tổng quan ACE

```
Training:
┌──────────────────────────────────────────────────────────┐
│  Input: zt (latent chứa target concept c)                │
│                                                          │
│  Conditional path:                                       │
│    ε(zt, c, t) → align với CEG → L_ESD                  │
│                                                          │
│  Unconditional path:                                     │
│    ε(zt, t) → align với PG-UEG → L_PUnc                 │
│                    ↑                                     │
│             (target guidance - prior correction)         │
│                                                          │
│  Prior preservation:                                     │
│    ε(zt, cp, t) → align với ε*_θ(zt, cp, t) → L_Cons    │
└──────────────────────────────────────────────────────────┘

Inference (CFG):
  ε_tilde = ε_θ(zt, t) + ω(ε_θ(zt, c_input, t) - ε_θ(zt, t))
              ↑
     Unconditional noise đã bị "nhiễm" erasure guidance
     → CFG tự động đẩy xa target concept bất kể c_input là gì
```

---

## 6. Kết quả trên Nudity Erasure

### 6.1 Setup đánh giá

- **Dataset:** I2P dataset (856 text prompts với nhãn explicit)
- **Detection tool:** NudeNet (đếm số body parts bị lộ)
- **Generation evaluation:** COCO-30k Caption (FID + CLIP score)
- **Robustness:** Adversarial attacks bằng UnlearnDiff, P4D, Ring-A-Bell
- **Editing evaluation:** SD-Inpainting với 200 ảnh, 20 editing texts (bao gồm adversarial text từ MMA-Diffusion)
- **Bounding:** GroundingDINO để detect clothing trong ảnh

### 6.2 Kết quả Exposure Detection trên I2P Dataset

> Số lượng body parts bị lộ trong ảnh được sinh ra (càng thấp càng tốt)

| Method | Buttocks↓ | Breast (F)↓ | Genitalia (F)↓ | Breast (M)↓ | Genitalia (M)↓ | Feet↓ | Armpits↓ | Belly↓ | **Total↓** | FID30k↓ | CLIP30k↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SD v1.4 | 61 | 204 | 37 | 38 | 16 | 70 | 241 | 183 | **850** | 14.07 | 0.313 |
| ESD | 15 | 29 | 5 | 11 | 10 | 37 | 68 | 36 | **211** | 13.80 | 0.304 |
| SPM | 14 | 29 | 7 | 2 | 12 | 41 | 53 | 28 | **186** | 14.63 | 0.312 |
| AdvUnlearn | 4 | 6 | 2 | 0 | 8 | 13 | 12 | 7 | **52** | 15.35 | 0.293 |
| MACE | 7 | 24 | 8 | 10 | 9 | 35 | 61 | 35 | **189** | 12.60 | 0.294 |
| RECE | 14 | 20 | 7 | 16 | 10 | 39 | 45 | 35 | **186** | 14.45 | 0.309 |
| **ACE (Ours)** | **3** | **2** | **3** | **4** | **9** | **6** | **5** | **7** | **39** | 14.69 | 0.308 |

**Nhận xét:**
- ACE đạt **total = 39**, giảm **95.4%** so với SD v1.4 gốc (850).
- ACE vượt trội AdvUnlearn (52) — đặc biệt ấn tượng vì AdvUnlearn dùng adversarial training chuyên dụng, còn ACE không.
- FID (14.69) và CLIP (0.308) của ACE gần với SD v1.4 gốc → model vẫn giữ được khả năng sinh ảnh chất lượng tốt sau khi xóa nudity.

### 6.3 Robustness trước Adversarial Attacks

> Attack Success Rate (ASR) — tỷ lệ attack thành công (càng thấp càng tốt)

| Method | UnlearnDiff↓ | P4D↓ | Ring-A-Bell↓ | **Average↓** |
|---|---|---|---|---|
| SD v1.4 | 100% | 100% | 85.21% | **95.07%** |
| ESD | 73.05% | 74.47% | 38.73% | **62.08%** |
| SPM | 91.49% | 91.49% | 57.75% | **80.24%** |
| AdvUnlearn | 25.53% | 19.15% | 4.93% | **16.54%** |
| MACE | 64.53% | 66.67% | 14.79% | **48.66%** |
| RECE | 70.92% | 65.96% | 26.76% | **54.55%** |
| **ACE (Ours)** | **27.65%** | **28.37%** | **2.82%** | **19.61%** |

**Nhận xét:**
- ACE đạt average ASR = **19.61%**, chỉ thua AdvUnlearn (16.54%) — nhưng AdvUnlearn dùng adversarial training còn ACE thì không.
- Đặc biệt trên Ring-A-Bell, ACE đạt **2.82%** — tốt nhất trong tất cả các phương pháp.
- ACE mạnh hơn đáng kể so với ESD (62%), SPM (80%), MACE (49%), RECE (55%).

### 6.4 Kết quả Explicit Editing (Nudity Editing Filtration)

> Số lượng nude detections trung bình trên mỗi 100 ảnh sau khi editing (càng thấp càng tốt)

| Method | Man↓ | Woman↓ | Overall↓ |
|---|---|---|---|
| Original (không edit) | 8 | 52 | 30 |
| SD v1.4 + editing | 51.75 | 110.60 | 81.18 |
| SPM | 25 | 86 | 55.5 |
| AdvUnlearn | 11.85 | 63.15 | 37.5 |
| **ACE (Ours)** | **12.80** | **66.85** | **39.83** |

**Nhận xét:**
- ACE đạt overall = **39.83**, gần bằng AdvUnlearn (37.5) mà không cần adversarial training.
- So với SD v1.4 thuần (81.18), ACE giảm ~51% số lượng nude detections sau editing.
- ACE là phương pháp duy nhất không dùng adversarial training mà vẫn đạt hiệu quả editing filtration cạnh tranh được với AdvUnlearn.

### 6.5 Tóm tắt hiệu quả trên Nudity Task

| Tiêu chí | ACE | Phương pháp tốt nhất khác |
|---|---|---|
| Generation prevention (Total↓) | **39** | 52 (AdvUnlearn) |
| Adversarial robustness (Avg ASR↓) | 19.61% | **16.54%** (AdvUnlearn) |
| Editing filtration (Overall↓) | **39.83** | 37.5 (AdvUnlearn) |
| FID (image quality)↓ | 14.69 | **12.60** (MACE) |
| CLIP score↑ | 0.308 | **0.312** (SPM) |

ACE đạt **top-2 trên hầu hết các tiêu chí** mà **không cần adversarial training** — điều làm ACE nổi bật về tính thực tiễn và hiệu quả.

---

## 7. Phân tích Ablation

| Variant | Editing Filtration (CLIPe↓) | Editing Filtration (LPIPSe↑) | Prior Preservation (CLIPp↑) |
|---|---|---|---|
| (1) Baseline (ESD only) | 0.301 | 0.060 | 0.305 |
| (2) + Unconditional (UEG) | 0.285 | 0.149 | 0.305 |
| (3) + UEG + LCons | 0.274 | 0.168 | 0.300 |
| (4) Full ACE (+ PG-UEG) | **0.274** | **0.168** | **0.303** |

- **UEG** cải thiện đáng kể editing filtration (+0.089 LPIPSe).
- **LCons** tăng cường khả năng xóa khái niệm nhưng gây một ít concept erosion.
- **PG-UEG** (correction guidance) phục hồi prior preservation mà không ảnh hưởng editing filtration.

---

## 8. ACE Diverse (biến thể mở rộng)

> **Code:** `src/train_ace_diverse.py` (dẫn xuất từ `src/train_ace.py`)

### 8.1 Động lực

ACE gốc xóa **một** chuỗi concept $c$ cố định (vd "nudity"). Tuy nhiên một concept có thể xuất hiện dưới **nhiều cách diễn đạt** khác nhau (khác ngữ cảnh, thuộc tính, văn phong). Nếu chỉ train trên một câu, erasure dễ bị bypass bởi các prompt diễn đạt cùng concept theo cách khác.

**ACE Diverse** thay khái niệm đơn $c$ bằng một **tập prompt đa dạng** đọc từ file CSV:

$$\mathcal{P} = \{p_1, p_2, \dots, p_K\}, \qquad p_k = c_{a,k} \oplus c$$

trong đó mỗi $p_k$ là sự kết hợp giữa concept mục tiêu $c$ với một ngữ cảnh/thuộc tính $c_{a,k}$ (attribute/context). Mục tiêu là phủ erasure lên **toàn bộ phân bố các cách diễn đạt** của $c$ thay vì một điểm duy nhất.

Mỗi iteration sample đồng đều một prompt:

$$p \sim \text{Uniform}(\mathcal{P})$$

### 8.2 Loss ESD (thay đổi) — $\mathcal{L}_{ESD}^{div}$

Thay vì xóa concept $c$, ta xóa prompt $p$ (chứa $c_a + c$) được sample ở mỗi bước:

$$\tilde{\epsilon}_p = \epsilon_{\theta^*}(z_t, t) - \eta_c \left(\epsilon_{\theta^*}(z_t, p, t) - \epsilon_{\theta^*}(z_t, t)\right)$$

$$\mathcal{L}_{ESD}^{div} = \mathbb{E}_{z_t, t,\; p \sim \mathcal{P}} \left\| \epsilon_\theta(z_t, p, t) - \tilde{\epsilon}_p \right\|_2^2 \tag{13}$$

So với Eq. (12), điểm khác duy nhất là điều kiện text $c \rightarrow p$ được lấy ngẫu nhiên từ $\mathcal{P}$ mỗi iteration. Qua nhiều bước, model học đẩy conditional noise ra xa **mọi phrasing** trong $\mathcal{P}$.

### 8.3 Loss Unconditional (giữ nguyên cấu trúc) — $\mathcal{L}_{PUnc}$

Giữ nguyên công thức PG-UEG (Eq. 8–9). Cấu trúc loss không đổi; chỉ khác ở chỗ latent $z_t$ và hướng target được sinh theo prompt $p$ đang được sample:

$$\tilde{\epsilon}^p_u = \epsilon_{\theta^*}(z_t, t) + \eta_u \left(\epsilon_{\theta^*}(z_t, p, t) - \epsilon_{\theta^*}(z_t, t)\right) - \eta_p \gamma_p \left(\epsilon_{\theta^*}(z_t, c_p, t) - \epsilon_{\theta^*}(z_t, t)\right)$$

$$\mathcal{L}_{PUnc}^{div} = \mathbb{E}_{z_t, t,\; p \sim \mathcal{P},\; c_p \in C_p} \left\| \epsilon_\theta(z_t, t) - \tilde{\epsilon}^p_u \right\|_2^2 \tag{14}$$

trong đó $\gamma_p = \dfrac{\text{CLIP}(x, c_p)}{\text{CLIP}(x, p)}$ — mẫu số được chuẩn hóa theo prompt $p$ hiện tại (yêu cầu file CLIP score `sc_clip` phải có entry cho **mọi** prompt trong $\mathcal{P}$).

### 8.4 Loss Preservation (giữ nguyên) — $\mathcal{L}_{Cons}$

Bảo tồn prior concept như ACE gốc (Eq. 10):

$$\mathcal{L}_{Cons} = \mathbb{E}_{z_t, t, c_p \in C_p} \left\| \epsilon_\theta(z_t, c_p, t) - \epsilon_{\theta^*}(z_t, c_p, t) \right\|_2^2$$

Lưu ý: tập prior $C_p$ được lọc để loại bỏ **tất cả** prompt thuộc $\mathcal{P}$ (điều kiện `not in concept`), tránh rò rỉ giữa tập erase và tập preserve.

### 8.5 Objective tổng hợp

$$\mathcal{L}_{ACE}^{div} = \lambda_{PUnc} \mathcal{L}_{PUnc}^{div} + \lambda_{Cons} \mathcal{L}_{Cons} + \lambda_{ESD} \mathcal{L}_{ESD}^{div}$$

### 8.6 Tóm tắt khác biệt so với ACE gốc

| Thành phần | ACE | ACE Diverse |
|---|---|---|
| Target concept | 1 chuỗi $c$ cố định | tập prompt $\mathcal{P} = \{c_{a,k} \oplus c\}$ từ CSV |
| Sampling mỗi bước | luôn dùng $c$ | $p \sim \text{Uniform}(\mathcal{P})$ |
| $\mathcal{L}_{ESD}$ | xóa $c$ (Eq. 12) | xóa $p$ (Eq. 13) — **thay đổi** |
| $\mathcal{L}_{PUnc}$ | Eq. 9 | Eq. 14 — **giữ cấu trúc**, target theo $p$ |
| $\mathcal{L}_{Cons}$ | Eq. 10 | **giữ nguyên**, $C_p$ loại bỏ toàn bộ $\mathcal{P}$ |
| CLIP normalization $\gamma_p$ | theo $c$ (cố định) | theo $p$ (thay đổi mỗi bước) |

**Lợi ích:** erasure tổng quát hơn, bền hơn trước nhiều cách diễn đạt của concept.
**Lưu ý triển khai:** file `sc_clip` json cần chứa CLIP score cho mọi prompt trong $\mathcal{P}$; nên kiểm soát giá trị $\gamma_p$ (clamp / đảm bảo CLIP score hợp lý) để tránh dao động mạnh khi mẫu số $\text{CLIP}(x, p)$ nhỏ.
