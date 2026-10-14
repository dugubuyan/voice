"""网页演示（可整体删除）。

约定：
- 依赖方向单向：demo -> app / seed；app / seed / tests 不得 import demo。
- 不修改 app/ 任何代码、不写入主库 elder_care.db（每位访客用独立的内存库沙箱）。
- 删除本目录即可完全移除演示，见 demo/README.md。
"""
