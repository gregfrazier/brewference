import groovy.json.JsonOutput
import java.time.LocalDate
import java.time.format.DateTimeFormatter

def out = new StringBuilder()
def jinjaLength = { value -> value == null ? 0 : value.size() }
def jinjaFirst = { value -> value == null || value.isEmpty() ? null : value[0] }
def jinjaLast = { value -> value == null || value.isEmpty() ? null : value[value.size() - 1] }
def jinjaToJson = { value -> JsonOutput.toJson(value) }
def jinjaTrim = { value -> value == null ? null : value.toString().trim() }
def jinjaStringValue = { value -> value == null ? null : value.toString() }
def jinjaStrftimeNow = { value -> LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMMM yyyy")) }
def jinjaSlice = { value, start, end, step ->
    if (value == null) return null
    int length = value.size()
    int stepSize = step == null ? 1 : step as int
    if (stepSize == 0) throw new IllegalArgumentException("slice step cannot be zero")
    int from = stepSize > 0 ? (start == null ? 0 : start as int) : (start == null ? length - 1 : start as int)
    int to = stepSize > 0 ? (end == null ? length : end as int) : (end == null ? -1 : end as int)
    if (from < 0) from += length
    if (to < 0 && (stepSize > 0 || to != -1)) to += length
    from = stepSize > 0 ? Math.max(0, Math.min(from, length)) : Math.max(-1, Math.min(from, length - 1))
    to = stepSize > 0 ? Math.max(0, Math.min(to, length)) : to
    def result = []
    for (int i = from; stepSize > 0 ? i < to : i > to; i += stepSize) result << value[i]
    return result
}
if (tools) {
def __jinjaCollection0 = tools
for (int __jinjaIndex0 = 0; __jinjaIndex0 < __jinjaCollection0.size(); __jinjaIndex0++) {
    def tool = __jinjaCollection0[__jinjaIndex0]
    def __jinjaLoop0 = [index0: __jinjaIndex0, index: __jinjaIndex0 + 1, first: __jinjaIndex0 == 0, last: __jinjaIndex0 == __jinjaCollection0.size() - 1]
out << "TOOL "
out << (jinjaToJson(tool))
out << "\n"
}
}
def __jinjaCollection1 = messages
for (int __jinjaIndex1 = 0; __jinjaIndex1 < __jinjaCollection1.size(); __jinjaIndex1++) {
    def message = __jinjaCollection1[__jinjaIndex1]
    def __jinjaLoop1 = [index0: __jinjaIndex1, index: __jinjaIndex1 + 1, first: __jinjaIndex1 == 0, last: __jinjaIndex1 == __jinjaCollection1.size() - 1]
out << (bos_token)
out << (message['role'])
out << "\n"
out << (message['content'])
out << "\n"
}
if (add_generation_prompt) {
out << (bos_token)
out << "model\n"
}
out << "\n"
out.toString()
