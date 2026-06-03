import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# This will trigger database loading, which is fine for a one-off diagnostics
from agent import LegalAgentOutput

json1 = """{
  "citations": null,
  "clarification_options": [
    "Hết hạn hợp đồng lao động",
    "Nhân viên đơn phương chấm dứt hợp đồng trái luật",
    "Thay đổi cơ cấu, công nghệ hoặc lý do kinh tế",
    "Kỷ luật lao động (sa thải)",
    "Người lao động thường xuyên không hoàn thành công việc"
  ],
  "clarification_question": "Để hỗ trợ bạn đúng quy định pháp luật lao động Việt Nam, bạn vui lòng cho biết lý do bạn muốn chấm dứt hợp đồng lao động với nhân viên này?",
  "detailed_analysis": null,
  "disclaimer": "Tuyên bố miễn trừ trách nhiệm: Thông tin do Trợ lý AI cung cấp chỉ mang tính chất tham khảo cho doanh nghiệp SME và không thay thế cho dịch vụ tư vấn pháp lý chuyên nghiệp của Luật sư.",
  "document_message": null,
  "download_url": null,
  "output_type": "clarification",
  "summary_answer": null
}"""

json2 = """{
  "citations": [
    {
      "url": "https://vbpl.vn/tw/Pages/vbpq-toanvan.aspx?ItemID=146643",
      "title": "Bộ luật Lao động 2019"
    }
  ],
  "clarification_question": null,
  "clarification_options": null,
  "detailed_analysis": "Theo Điều 27 Bộ luật Lao động 2019, trong thời gian thử việc, cả người lao động và người sử dụng lao động đều có quyền chấm dứt hợp đồng thử việc mà không cần báo trước và không phải bồi thường nếu việc làm thử không đạt yêu cầu như hai bên đã thỏa thuận. Nếu nội bộ không đạt mong muốn, bạn có thể thông báo chấm dứt thử việc cho người lao động ngay lập tức.",
  "document_message": null,
  "download_url": null,
  "output_type": "advice",
  "summary_answer": "Bạn có thể đơn phương chấm dứt hợp đồng thử việc ngay lập tức mà không cần báo trước nếu người lao động không đạt yêu cầu thử việc."
}"""

print("=== Validating JSON 1 ===")
try:
    obj1 = LegalAgentOutput.model_validate_json(json1)
    print("JSON 1 is VALID!")
    print(obj1.model_dump())
except Exception as e:
    print("JSON 1 validation FAILED:")
    import traceback
    traceback.print_exc()

print("\n=== Validating JSON 2 ===")
try:
    obj2 = LegalAgentOutput.model_validate_json(json2)
    print("JSON 2 is VALID!")
    print(obj2.model_dump())
except Exception as e:
    print("JSON 2 validation FAILED:")
    import traceback
    traceback.print_exc()
