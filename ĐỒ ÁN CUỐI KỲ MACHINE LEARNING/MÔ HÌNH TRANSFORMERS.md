# Giới thiệu
Mô hình Transformers thực hiện rất nhiều tasks khác nhau trong xử lý ngôn ngữ tự nhiên từ dịch máy, viết thơ, tranh luận, thậm chí là viết code. Ngoài ra còn được sử dụng trong sinh học để giải quyết bài toán cuộn gập protein,...
Các mô hình nổi tiếng như BERT, GPT-3, T5 đều dựa vào kiến trúc Transformers
# Lịch sử ra đời kiến trúc Transformers
Transformers là một dạng kiến trúc mạng lưới neuron.
Trước khi có Transformers, con người chưa có những mô hình huấn luyện tốt cho ngôn ngữ.
Họ sử dụng mô hình RNN (Recurrent Neural Network)
![[Pasted image 20230323163954.png]]
Ví dụ: Dịch một câu từ tiếng Anh sang tiếng Pháp
RNN nhận đầu vào là một câu tiếng Anh và xử lý từng từ cùng một thời điểm, sau đó tuần tự trả ra các từ tiếng Pháp tương ứng.
Quan trọng nhất là sự tuần tự (Sequential) vì trật từ các từ trong câu rất quan trọng, không thể sắp xếp lung tung
RNN xem xét từng từ một cách tuần tự

Nhưng RNN có những hạn chế như:
- Xử lý các chuỗi câu lớn, đoạn văn dài hoặc một bài luận: Khi nó xử lý phần kết của đoạn văn thì nó sẽ quên phần đầu của đoạn văn
- RNN rất khó để huấn luyện: Vì chúng xử lý một cách tuần tự, nên chúng không thể chạy song song, nên khó để tăng tốc độ

Để khắc phục thì mô hình Transformers ra đời. Các nghiên cứu sinh ở Google và đại học Toronto đã cho ra mắt mô hình Transformers vào năm 2017.
Mô hình Transformers được thiết kế với mục địch là dịch thuật. Khác với mô hình RNN, Trans có thể chạy song song các mô hình với nhau rất hiệu quả. Cho nên nếu sử dụng các phần cứng máy tính hiệu quả (CPU, GPU) thì có thể huấn luyện một mô hình rất lớn.
Ví dụ: Mô hình GPT-3 được huấn luyện với 45TB dữ liệu văn bản
# Kiến trúc Transformers là gì?
![[Pasted image 20230323170741.png]]
Có ba thành phần (cải tiến) chính trong bài báo khiến mô hình này hoạt động rất tốt
- Positional Encoding
- Attention
- Self-attention
## Positional Encoding
Ý tưởng là trước khi đưa vào mạng neuron, lấy từng từ trong câu và gắn chúng với một chữ số, đơn giản là lưu trữ theo dạng có thứ tự thay vì theo cấu trúc của một mạng. Và khi huấn luyện mạng neuron với dữ liệu văn bản lớn, nó có thể học được cách diễn giải các Positional Encoding.
Với cách này thì mạng neuron học được các trật tự từ dữ liệu. Điều này giúp Transformers huấn luyện tốt hơn RNN.
## Attention
Khái niệm này có khắp mọi nơi trong Machine Learning. Nên nhớ rằng mô hình Transformers ban đầu dùng để dịch thuật.
Cơ chế Attention là một cấu trúc mạng neuron cho phép mô hình văn bản kiểm tra từng từ đơn lẻ trong câu gốc khi đưa ra quyết định về cách dịch một từ ở đầu ra.
![[Pasted image 20230323210812.png]]
Các ô càng sáng biểu thị rằng 1 từ A từ ngôn ngữ E1 "chú ý" hay có tương quan hơn (correlation) với 1 từ B từ ngôn ngữ E2

### Vậy làm sao để mô hình biết từ nào nó nên "chú ý" nhiều hơn?
Mô hình cần học theo thời gian từ dữ liệu. Huấn luyện hàng ngàn ví dụ về các cặp câu, trật tự từ, giống,... và tất cả những thứ liên quan tới ngữ pháp.
## Self-Attention
Attention đã được phát minh từ trước khi có bài báo này. Mô hình Transformers nâng cấp lên thành Self-Attention, một bước ngoặt so với Attention truyền thống.

Việc căn chỉnh các từ giữa 2 tiếng khác nhau là một điều quan trọng trong việc dịch thuật. Nhưng nếu chỉ muốn biết nghĩa cơ bản trong ngôn ngữ mà có thể xây dựng nên một mạng lưới mà có thể thực hiện nhiều bài toán ngôn ngữ khác. 

Khi các mạng neuron phân tích rất nhiều dữ liệu văn bản, chúng tự động xây dựng nên Internal Representation, hoặc đơn giản là sự hiểu biết về ngôn ngữ. Ví dụ, chúng có thể học được các từ như Programmer, Software Engineer, hoặc Software Developer đều đồng nghĩa như nhau.
Chúng cũng có thể học một cách tự nhiên các quy tắc ngữ pháp, giống... Nên nếu như Internal Representation càng tốt thì chúng có thể thực hiện càng tốt hơn các bài toán ngôn ngữ khác.
**Vì vậy, Attention có thể là một cách rất hiệu quả để giúp mạng neuron hiểu về ngôn ngữ nếu nó được "chú ý" từ văn bản đầu vào.**
Ví dụ: Cho hai câu
"Server, can I have a **check**" và "Looks like I just **crashed** the server"
Hai từ server mang ý nghĩa khác nhau vì ta xem xét vào ngữ cảnh của các từ xung quanh

- Self-Attention cho phép mạng neuron hiểu được một từ trong ngữ cảnh của các từ xung quanh nó.
Ví dụ: Nếu mô hình xử lý từ "Server" ở câu đầu tiên, nó sẽ xem xét từ **"check"**, giúp nó phân biệt được người phục vụ và máy chủ
Với câu thứ hai, mô hình chú ý đến từ **"crashed"** để xác định từ "server" là máy chủ
- Self-Attention còn giúp mạng neuron phân biệt các từ, nhận dạng các đoạn của một đoạn lời nói, thậm chí là thì của các từ.
# Mô hình Transformers được sử dụng như nào?
Mô hình phổ biến nhất sử dụng Transformers có tên là BERT (Bidirectional Encoder Representations from Transformers), được Google công bố vào năm 2018.
BERT được huấn luyện dựa trên một tập dữ liệu văn bản khồng lồ, sử dụng rộng rãi trong NLP với các điều chỉnh cho phù hợp với một loạt các tasks khác nhau như Text Generation, Question Answering, Classification, Finding similar sentences. Mô hình này sử dụng trong Google Search để truy vấn các tìm kiếm và hỗ trợ nhiều cho Google Cloud.
BERT cũng có thể xây dựng các mô hình rất tốt trên tập dữ liệu không dán nhãn như văn bản từ Wikipedia hoặc Reddit, còn được gọi là Semi Supervised Learning. 
