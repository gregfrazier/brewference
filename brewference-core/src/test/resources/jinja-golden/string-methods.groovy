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
out << (line.stripTrailing())
out << "|"
out << (jinjaTrim(line.stripLeading()))
out << "|"
out << (padded.strip())
out << "|"
out << (name.startsWith("wor"))
out << "|"
out << (name.endsWith("ld"))
out << "\n"
out.toString()
