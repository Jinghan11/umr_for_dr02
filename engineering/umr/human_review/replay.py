#!/usr/bin/env python3
"""Synchronized original SOMA / native / contact-refined 29DOF motion review."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CACHE = HERE / "replay_cache"
CATEGORY_LABELS = {"locomotion": "行走／跑动／转向", "upper_body": "上肢动作", "dance_sport": "舞蹈／运动",
                   "jump_flight": "跳跃／腾空", "floor_transition": "地面／姿态过渡", "interaction": "交互", "other": "未归类"}
DECISION_LABELS = {"待填写": "", "通过": "PASS", "轻微问题": "MINOR", "明显问题": "MAJOR",
                   "严重错误": "CRITICAL", "无法判断": "UNSURE"}
ISSUE_LABELS = {"position": "关节位置／姿态", "velocity": "速度／平滑", "contact": "脚部离地／接触",
                "source": "源动作问题", "alignment": "时钟／坐标／映射"}
PHASE_LABELS = {"reviewing": "人工审核中", "batch_complete": "本批已提交", "ready_for_repair": "可生成处理清单",
                "repair_planned": "等待重处理产物", "repair_registered": "新产物已登记",
                "awaiting_agent_review": "等待 AI 对照", "agent_reviewed": "AI 已回答，待人工复核",
                "no_repair_needed": "本轮无处理目标", "followup_needed": "需要补充观察证据"}
ROLE_LABELS = {"coverage": "覆盖样本", "risk": "风险样本", "feedback_risk": "反馈推荐",
               "random_monitoring": "随机监测", "repaired_old": "修复后复核", "pending_recheck": "未决补证据",
               "unseen_exploration": "未看新数据", "old_pass_regression": "旧通过复看", "direct": "指定动作"}


def adaptive_module():
    try:
        from . import adaptive
    except ImportError:
        import adaptive
    return adaptive


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_items(run):
    """Use exactly the frozen queue, including its trajectory identity."""
    run = Path(run).resolve()
    for line in (run / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or sha(run / name) != expected:
            raise ValueError(f"冻结审核清单已改变：{name}")
    manifest = json.loads((run / "sample_manifest.json").read_text())
    items = manifest["items"]
    if not items or len({item["review_id"] for item in items}) != len(items):
        raise ValueError("审核清单为空或包含重复身份")
    return items


def choose_item(items, identity):
    if not identity:
        return items[0]
    matches = [r for r in items if identity in (r["review_id"], r["motion_id"], r["member"])]
    if len(matches) != 1:
        raise ValueError(f"审核动作不存在或身份不唯一：{identity}")
    return matches[0]


def csv_items(path):
    """Open a processing review list in its recorded order, without resampling it."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"动作名称", "状态", "结果文件夹"}.issubset(reader.fieldnames or ()):
            raise ValueError("清单缺少动作名称、状态或结果文件夹")
        rows = list(reader)
    items = []
    for row in rows:
        motion = Path(row["结果文件夹"]).expanduser()
        if not motion.is_absolute():
            motion = path.parent / motion
        motion = motion.resolve()
        contact = json.loads((motion / "contact_report.json").read_text())
        provenance = json.loads((motion / "provenance.json").read_text())
        native = json.loads((motion / "native_identity.json").read_text())
        trajectory = motion / "trajectory.npz"
        if not trajectory.is_file() or not (motion / "native_trajectory.npz").is_file():
            raise ValueError(f'动作缺少已保存轨迹：{row["动作名称"]}')
        if motion.name != row["动作名称"]:
            raise ValueError(f'动作名称与结果目录不一致：{row["动作名称"]}')
        items.append({"review_id": provenance["source"]["key"], "motion_id": row["动作名称"],
                      "member": provenance["source"]["member"], "trajectory": str(trajectory),
                      "native_trajectory": str(motion / "native_trajectory.npz"),
                      "contact_report": str(motion / "contact_report.json"),
                      "final_sha256": contact["trajectory_sha256"], "native_sha256": native["sha256"],
                      "model_sha256": contact["robot_xml_sha256"], "sampling_roles": "direct",
                      "processing_status": row["状态"], "quality_hint": row.get("质量提示", ""),
                      "source_csv": str(path)})
    if not items or len({item["review_id"] for item in items}) != len(items):
        raise ValueError("清单为空或包含重复动作")
    return items


def prepare(item, cache, *, source_mode="skeleton"):
    try:
        from .replay_data import load_replay
    except ImportError:
        from replay_data import load_replay
    trajectory = Path(item["trajectory"]).resolve()
    # Validate the actual bytes against this frozen sample, in addition to the
    # report / native / model / original SOMA checks performed by load_replay.
    if item.get("final_sha256") and sha(trajectory) != item["final_sha256"]:
        raise ValueError("最终轨迹与冻结审核清单指纹不符")
    result = load_replay(trajectory.parent, Path(cache).resolve(), source_mode=source_mode)
    if len(result.native_qpos) != len(result.final_qpos):
        raise ValueError("原生/修正轨迹帧数不一致")
    return result


def replay_summary(data, item):
    return {"motion_id": item["motion_id"], "review_id": item["review_id"],
            "frames": len(data.final_qpos), "fps": float(data.fps),
            "native_shape": list(data.native_qpos.shape), "final_shape": list(data.final_qpos.shape),
            "source_vertices_shape": list(data.source_vertices.shape),
            "source_faces_shape": list(data.source_faces.shape),
            "model_path": str(data.model_path), "metadata": data.metadata,
            "source_kind": getattr(data, "source_kind", "mesh"),
            "human_review_completed": False}


@dataclass
class PlaybackClock:
    frames: int
    fps: float = 50.0
    position: float = 0.0

    def seek(self, frame):
        self.position = float(max(0, min(self.frames - 1, int(frame))))
        return int(self.position)

    def advance(self, elapsed, speed=1.0, loop=False):
        if not np.isfinite(elapsed) or elapsed < 0 or not np.isfinite(speed) or speed <= 0:
            raise ValueError("播放时间与速度必须为有效正值")
        position = self.position + elapsed * self.fps * speed
        ended = position >= self.frames
        if ended:
            position = position % self.frames if loop else float(self.frames - 1)
        # Keep sub-frame time through the final sample interval. Clamping on
        # every tick discards it and can strand fast or slow playback here.
        self.position = position
        return int(self.position), ended and not loop


class ReviewViewer:
    def __init__(self, args, items):
        import viser
        try:
            from .replay_buffer import ReplayBuffer
        except ImportError:
            from replay_buffer import ReplayBuffer
        self.args, self.items = args, items
        self.events = queue.SimpleQueue()
        self.data = self.scene = self.clock = None
        self.loaded_item = None
        self.loading = False
        self.playing = False
        self.looping = False
        self.closed = False
        self.visible_items = items
        self.replay_buffer = ReplayBuffer(lambda item: prepare(item, args.cache, source_mode="skeleton"), max_bytes=256 * 1024 ** 2)
        self.soma_buffer = ReplayBuffer(lambda item: prepare(item, args.cache, source_mode="mesh"), max_bytes=2 * 1024 ** 3)
        self.last_switch_seconds = None
        # Viser starts a background listener. Fail promptly if the requested
        # port cannot be opened, rather than waiting inside server startup.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
        self.server = viser.ViserServer(host=args.host, port=args.port, label="重定向动作审核 · 29DOF")
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large",
                                        brand_color=(53, 115, 206), show_share_button=False)
        gui = self.server.gui
        gui.add_markdown("## 重定向动作审核")
        self.pipeline_note = gui.add_markdown("")
        if getattr(args, "review_csv", None):
            self.pipeline_note.content = f"指定清单 · {len(items)} 条"
        with gui.add_folder("审核动作"):
            self.search = gui.add_text("搜索动作", initial_value="")
            categories = sorted({item.get("category", "other") for item in items})
            self.category = gui.add_dropdown("动作类别", ["全部"] + [CATEGORY_LABELS.get(c, c) for c in categories])
            self.motion = gui.add_dropdown("动作", self.labels(items))
            self.load = gui.add_button("加载所选动作", color="blue")
            self.previous = gui.add_button("上一条动作")
            self.next = gui.add_button("下一条动作")
        self.status = gui.add_markdown("正在准备动作…")
        self.preload_note = gui.add_markdown("")
        with gui.add_folder("同步回放"):
            self.playback_buttons = gui.add_button_group("回放", ["暂停", "循环：关"])
            self.speed = gui.add_dropdown("播放速度", ["0.25×", "0.5×", "1×", "2×"], initial_value="1×")
            self.frame = gui.add_slider("共同帧", 0, 1, 1, 0, disabled=True)
            self.previous_frame = gui.add_button("前一帧")
            self.next_frame = gui.add_button("后一帧")
            self.first_frame = gui.add_button("回到首帧")
            self.last_frame = gui.add_button("跳到末帧")
            self.progress = gui.add_markdown("")
            self.source_status = gui.add_markdown("源动作：BVH 骨架")
        with gui.add_folder("动作信息", expand_by_default=False):
            self.details = gui.add_markdown("")
        if args.session:
            with gui.add_folder("单人审核记录"):
                self.decision = gui.add_dropdown("本条结论", list(DECISION_LABELS))
                self.issue_handles = {key: gui.add_checkbox(label, False) for key, label in ISSUE_LABELS.items()}
                self.body_parts = gui.add_text("涉及部位", initial_value="", hint="例如左腿、右脚、腰部；多个部位可用逗号分隔")
                self.event_start = gui.add_number("问题开始帧", -1, min=-1, step=1, hint="帧号从0开始，-1表示未指定")
                self.event_end = gui.add_number("问题结束帧", -1, min=-1, step=1, hint="包含结束帧，-1表示未指定")
                self.review_notes = gui.add_text("问题描述", initial_value="", multiline=True)
                self.save = gui.add_button("保存本条审核", color="blue")
                self.save_next = gui.add_button("保存并下一条")
                self.review_status = gui.add_markdown("")
            with gui.add_folder("反馈与下一批"):
                self.refresh_batch = gui.add_button("刷新当前审核批次")
                self.recommend_more = gui.add_button("根据反馈推荐下一批")
                self.repair_plan = gui.add_button("生成本轮重处理清单")
                self.run_ai = gui.add_button("对新产物运行 AI 前后对照")
                self.mixed_batch = gui.add_button("创建新旧混合复核批次")
            self.save.on_click(lambda _: self.events.put(("save_review", False)))
            self.save_next.on_click(lambda _: self.events.put(("save_review", True)))
            self.refresh_batch.on_click(lambda _: self.events.put(("refresh_batch", None)))
            self.recommend_more.on_click(lambda _: self.events.put(("pipeline", "recommend")))
            self.repair_plan.on_click(lambda _: self.events.put(("pipeline", "plan_repair")))
            self.run_ai.on_click(lambda _: self.events.put(("pipeline", "agent_review")))
            self.mixed_batch.on_click(lambda _: self.events.put(("pipeline", "next_cycle")))
            self.refresh_pipeline()
        gui.add_markdown("拖动画面可旋转视角。短促异常可用慢放或逐帧检查；审核结论需由实际观看者填写。")
        with gui.add_folder("观察方式", expand_by_default=False):
            self.layout = gui.add_dropdown("显示方式", ["并排", "叠加"])
            self.separation = gui.add_slider("并排间距（米）", .5, 5., .25, 2.)
            self.show_source = gui.add_checkbox("显示源动作", True)
            self.soma_body = gui.add_checkbox("SOMA 人体（按需加载）", False)
            self.show_native = gui.add_checkbox("显示原生结果", True)
            self.show_final = gui.add_checkbox("显示修正结果", True)
            self.source_opacity = gui.add_slider("SOMA 人体透明度", .1, 1., .1, .8, disabled=True)
            self.native_opacity = gui.add_slider("原生结果透明度", .1, 1., .1, 1.)
            self.final_opacity = gui.add_slider("修正结果透明度", .1, 1., .1, 1.)
            self.view = gui.add_dropdown("预设视角", ["自由视角", "正面", "侧面", "俯视"], initial_value="正面")
            self.camera_reset = gui.add_button("重置视角")
            self.follow = gui.add_checkbox("相机跟随位移", False)
        self.motion.on_update(lambda e: None)
        for handle in [self.search, self.category]:
            handle.on_update(lambda e: self.events.put(("filter", None)) if e.client is not None else None)
        self.playback_buttons.on_click(lambda e: self.events.put(("playback", e.target.value)))
        self.load.on_click(lambda _: self.events.put(("load", None)))
        self.previous.on_click(lambda _: self.events.put(("adjacent", -1)))
        self.next.on_click(lambda _: self.events.put(("adjacent", 1)))
        self.frame.on_update(lambda e: self.events.put(("seek", int(e.target.value))) if e.client is not None else None)
        self.previous_frame.on_click(lambda _: self.events.put(("step", -1)))
        self.next_frame.on_click(lambda _: self.events.put(("step", 1)))
        self.first_frame.on_click(lambda _: self.events.put(("seek", 0)))
        self.last_frame.on_click(lambda _: self.events.put(("seek", -1)))
        for handle in [self.layout, self.separation, self.show_source, self.show_native, self.show_final,
                       self.source_opacity, self.native_opacity, self.final_opacity]:
            handle.on_update(lambda _: self.events.put(("render", None)))
        self.view.on_update(lambda e: self.events.put(("camera", None)) if e.client is not None else None)
        self.camera_reset.on_click(lambda _: self.events.put(("camera", None)))
        self.soma_body.on_update(lambda e: self.events.put(("source_mode", bool(e.target.value))) if e.client is not None else None)
        self.server.on_client_connect(lambda client: self.events.put(("client", client)))
        initial = choose_item(items, args.motion_id)
        self.motion.value = self.label(initial)
        self.start_loading(initial)
        if getattr(args, "review_csv", None):
            # Selected BVHs share one archive pass, outside the foreground
            # motion loader. No SOMA bodies or review decisions are generated.
            def cache_originals():
                try:
                    try:
                        from .replay_data import prepare_raw_sources
                    except ImportError:
                        from replay_data import prepare_raw_sources
                    result = prepare_raw_sources([Path(item["trajectory"]).parent for item in items], args.cache)
                    print(json.dumps({"event": "csv_bvh_cache_ready", "selected": result["selected"]}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    print(json.dumps({"event": "csv_bvh_cache_error", "error": str(exc)}, ensure_ascii=False), flush=True)
            threading.Thread(target=cache_originals, name="csv-bvh-cache", daemon=True).start()

    @staticmethod
    def label(item):
        roles = item.get("recommendation_role", item.get("sampling_roles", "direct"))
        roles = roles if isinstance(roles, list) else [roles]
        label = "／".join(ROLE_LABELS.get(role, role) for role in roles)
        return f'{item["motion_id"]} · {label} · {item["review_id"][:8]}'

    @classmethod
    def labels(cls, items):
        return [cls.label(item) for item in items]

    def selected(self):
        return next((item for item in self.visible_items if self.label(item) == self.motion.value), None)

    def filter(self):
        text = self.search.value.strip().casefold()
        self.visible_items = [item for item in self.items
                              if (self.category.value == "全部" or CATEGORY_LABELS.get(item.get("category"), item.get("category")) == self.category.value)
                              and (not text or text in (item["motion_id"] + " " + item["review_id"]).casefold())]
        old = self.motion.value
        labels = self.labels(self.visible_items) or ["没有匹配动作"]
        self.motion.options = labels
        self.motion.value = old if old in labels else labels[0]
        self.motion.disabled = not bool(self.visible_items)
        self.load.disabled = self.loading or not bool(self.visible_items)
        self.prefetch_adjacent()

    def start_loading(self, item):
        if self.loading or item is None:
            return
        self.loading = True
        self.set_playing(False)
        self.load.disabled = self.previous.disabled = self.next.disabled = True
        self.load_started_at = time.monotonic()
        self.status.content = f'正在读取 **{item["motion_id"]}** 的 BVH 骨架和机器人轨迹…'
        future = self.replay_buffer.request(item, retry=True)
        def completed(result):
            try:
                self.events.put(("loaded", (item, result.result())))
            except Exception as exc:
                self.events.put(("load_error", str(exc)))
        future.add_done_callback(completed)

    def install(self, item, data):
        try:
            from .replay_data import file_sha256
        except ImportError:
            from replay_data import file_sha256
        if data.metadata.get("robot_xml_sha256") != file_sha256(data.model_path):
            raise ValueError("机器人模型与已验证回放不一致")
        try:
            from .replay_scene import ReplayScene
        except ImportError:
            from replay_scene import ReplayScene
        reused = self.scene is not None and hasattr(self.scene, "set_replay") and self.scene.set_replay(data)
        if not reused:
            if self.scene is not None:
                self.scene.close()
            self.scene = ReplayScene(self.server, data)
        self.data, self.loaded_item = data, item
        self.clock = PlaybackClock(len(data.final_qpos), data.fps)
        self.frame.max = len(data.final_qpos) - 1
        if self.args.session:
            self.event_start.max = self.event_end.max = self.frame.max
        self.frame.disabled = False
        self.loading = False
        self.load.disabled = self.previous.disabled = self.next.disabled = False
        context = "清单预览" if getattr(self.args, "review_csv", None) else "单人审核"
        self.status.content = f'当前画面：**{item["motion_id"]}** · {context} · 三路同步 {data.fps:g} Hz'
        rejected = data.metadata.get("contact_accepted") is False
        self.scene.final_label.text = "修正候选（未通过）· 蓝色 · 29 自由度" if rejected else "修正结果 · 蓝色 · 29 自由度"
        self.details.content = (f'类别：{CATEGORY_LABELS.get(item.get("category"), item.get("category", "—"))}  \n风险：{item.get("risk", "—")}（待校准）  \n'
                                f'帧数：{len(data.final_qpos)}  \n机器人：29 自由度  \n'
                                f'来源：BONES-SEED 原始 SOMA  \n审核身份：`{item["review_id"]}`  \n'
                                '原生与修正结果使用同一保存模型；原轨迹保持原样。')
        if self.args.session:
            self.fill_review(item)
            self.details.content += f'  \n推荐原因：{item.get("recommendation_reason", "—")}'
        if item.get("source_csv"):
            self.pipeline_note.content = f'指定清单 · {self.items.index(item) + 1}/{len(self.items)}'
            state = "接触修正未通过，显示已保存候选" if rejected else "处理完成，待目视查看"
            hint = item.get("quality_hint", "").replace("large_root_correction", "根部修正过大").replace(
                "foot_slip_contact_speed_p95_over_0_10_m_s", "接触期脚部滑动偏大")
            self.status.content += f'  \n{state}' + (f' · {hint}' if hint else "")
            self.details.content += f'  \n清单：{Path(item["source_csv"]).name}  \n处理状态：{item["processing_status"]}'
        self.render()
        self.camera()
        self.last_switch_seconds = time.monotonic() - self.load_started_at
        self.prefetch_adjacent()
        self.update_preload_status()
        if self.soma_body.value:
            self.request_soma()
        else:
            self.source_status.content = "源动作：BVH 骨架"
        self.set_playing(True)
        print(json.dumps({"event": "motion_ready", **replay_summary(data, item)}, ensure_ascii=False), flush=True)

    def prefetch_adjacent(self):
        if self.loaded_item and self.loaded_item in self.visible_items:
            self.replay_buffer.prefetch(self.visible_items, self.visible_items.index(self.loaded_item), ahead=3, behind=2)

    def update_preload_status(self):
        if not self.loaded_item or self.loaded_item not in self.visible_items:
            return
        index = self.visible_items.index(self.loaded_item)
        elapsed = "" if self.last_switch_seconds is None else f"本次切换 {self.last_switch_seconds:.2f} 秒 · "
        if index + 1 < len(self.visible_items):
            status = self.replay_buffer.status(self.visible_items[index + 1])
            label = {"ready": "已就绪", "loading": "后台准备中", "queued": "等待后台准备",
                     "error": "准备失败，可点击重试", "evicted": "已存入磁盘缓存"}.get(status["state"], "尚未准备")
            self.preload_note.content = f"{elapsed}下一条骨架：{label}"
        else:
            self.preload_note.content = elapsed.rstrip(" · ") + " · 已到当前队列末条"

    def request_soma(self):
        if not self.loaded_item or self.loading or not self.soma_body.value:
            return
        item = self.loaded_item
        self.source_status.content = "SOMA 人体正在后台准备，当前继续显示 BVH 骨架。"
        self.source_opacity.disabled = True
        future = self.soma_buffer.request(item, retry=True)
        def completed(result):
            try:
                self.events.put(("soma_ready", (item, result.result())))
            except Exception as exc:
                self.events.put(("soma_error", (item, str(exc))))
        future.add_done_callback(completed)

    def update_playback_buttons(self):
        self.playback_buttons.options = ("暂停" if self.playing else "播放", "循环：开" if self.looping else "循环：关")

    def set_playing(self, playing):
        self.playing = bool(playing)
        if self.playing:
            self.playback_started_at = time.monotonic()
        self.update_playback_buttons()

    def render(self):
        if self.scene is None or self.clock is None:
            return
        index = int(self.clock.position)
        self.scene.set_frame(index, side_by_side=self.layout.value == "并排", separation=float(self.separation.value),
                             source_opacity=float(self.source_opacity.value), native_opacity=float(self.native_opacity.value),
                             final_opacity=float(self.final_opacity.value), show_source=self.show_source.value,
                             show_native=self.show_native.value, show_final=self.show_final.value)
        if self.follow.value and hasattr(self.scene, "follow_camera"):
            self.scene.follow_camera()
        self.frame.value = index
        raw = float(self.data.source_frame_ids[index])
        raw_label = str(int(raw)) if raw.is_integer() else f"{raw:.3f}"
        self.progress.content = (f'**帧 {index} / {self.clock.frames - 1}** · {index / self.clock.fps:.3f} 秒  \n'
                                 f'原始采集帧：{raw_label} · 三路使用同一帧索引')

    def camera(self, client=None):
        if self.scene:
            view = {"自由视角": "perspective", "正面": "front", "侧面": "side", "俯视": "top"}[self.view.value]
            if hasattr(self.scene, "display_axis"):
                self.scene.display_axis = "x" if view == "side" else "y"
                self.render()
            self.scene.set_camera(view=view, client=client)

    def fill_review(self, item):
        review = item.get("review", {})
        self.decision.value = next((label for label, code in DECISION_LABELS.items() if code == review.get("decision", "")), "待填写")
        issues = review.get("issue_types", [])
        for key, handle in self.issue_handles.items():
            handle.value = key in issues
        parts = review.get("body_parts") or review.get("joints", [])
        self.body_parts.value = ",".join(parts) if isinstance(parts, list) else parts
        self.review_notes.value = review.get("notes", "")
        for endpoint, handle in (("start", self.event_start), ("end", self.event_end)):
            frame = review.get(f"event_{endpoint}_frame")
            seconds = review.get(f"event_{endpoint}_s")
            handle.value = int(frame) if frame is not None else round(float(seconds) * self.data.fps) if seconds is not None else -1
        self.review_status.content = "已读取该动作的审核记录。" if review else "本条尚未填写审核结论。"

    def refresh_pipeline(self):
        status = adaptive_module().session_status(self.args.session)
        count = status["recommendation"]["count"]
        summary = [f'**第 {status["cycle"]} 轮 · 已审 {status["completed"]}/{status["total"]}**']
        if status["unsure"]:
            summary.append(f'待定 {status["unsure"]}')
        if status.get("pending_backlog_count", 0):
            summary.append(f'未决 {status["pending_backlog_count"]}')
        if status["phase"] != "reviewing":
            summary.append(f'建议补荐 {count} 条' if count and status["phase"] == "batch_complete"
                           else PHASE_LABELS.get(status["phase"], status["phase"]))
        self.pipeline_note.content = " · ".join(summary)
        self.recommend_more.disabled = self.loading or count is None
        self.repair_plan.disabled = self.loading or status["pending"] > 0
        self.run_ai.disabled = self.loading or status["phase"] not in {"repair_registered", "awaiting_agent_review"}
        self.mixed_batch.disabled = self.loading or status["phase"] not in {"agent_reviewed", "no_repair_needed", "followup_needed"}
        return status

    def save_current_review(self, go_next=False):
        if not self.args.session or not self.loaded_item or self.loading:
            return
        decision = DECISION_LABELS[self.decision.value]
        review = adaptive_module().save_ui_review(
            self.args.session, self.loaded_item["review_id"], decision,
            [key for key, handle in self.issue_handles.items() if handle.value],
            notes=self.review_notes.value, body_parts=self.body_parts.value,
            event_start_frame=None if self.event_start.value == -1 else self.event_start.value,
            event_end_frame=None if self.event_end.value == -1 else self.event_end.value,
            fps=self.data.fps, frame_count=self.clock.frames)
        self.loaded_item["review"] = review
        self.review_status.content = "审核记录已保存。"
        self.refresh_pipeline()
        if go_next:
            # The dropdown may have changed without loading it. Advance from
            # the motion actually watched and just saved.
            if self.loaded_item in self.visible_items:
                self.motion.value = self.label(self.loaded_item)
            self.event("adjacent", 1)

    def reload_batch(self):
        if not self.args.session or self.loading:
            return
        items = adaptive_module().load_batch(self.args.session)
        self.items = self.visible_items = items
        self.search.value = ""
        self.category.options = ["全部"] + [CATEGORY_LABELS.get(c, c) for c in sorted({r.get("category", "other") for r in items})]
        self.category.value = "全部"
        self.motion.options = self.labels(items)
        self.motion.value = self.label(items[0])
        self.refresh_pipeline()
        self.start_loading(items[0])

    def pipeline_action(self, action):
        if self.loading or not self.args.session:
            return
        self.loading = True
        self.recommend_more.disabled = self.repair_plan.disabled = self.run_ai.disabled = self.mixed_batch.disabled = True
        self.pipeline_note.content = "正在根据本轮反馈更新工作清单…"
        def worker():
            try:
                if action == "agent_review":
                    try:
                        from . import dispatch_agent
                    except ImportError:
                        import dispatch_agent
                    result = dispatch_agent.dispatch_session(self.args.session,
                        progress=lambda message: self.events.put(("pipeline_progress", message)))
                    if not result.get("completed"):
                        raise ValueError(f'AI 对照未完成：{result.get("status")}; 记录：{result.get("attempt", self.args.session)}')
                elif action == "next_cycle":
                    status = adaptive_module().session_status(self.args.session)
                    count = (status.get("agent_review") or {}).get("next_review_count", 50)
                    result = adaptive_module().next_cycle(self.args.session, 8, 8, 4) if count == 20 else adaptive_module().next_cycle(self.args.session)
                else:
                    result = getattr(adaptive_module(), action)(self.args.session)
                self.events.put(("pipeline_result", (action, result)))
            except Exception as exc:
                self.events.put(("pipeline_error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def event(self, name, value):
        if name == "source_mode":
            if value:
                self.request_soma()
            elif self.scene:
                self.scene.set_source_kind("skeleton")
                self.source_opacity.disabled = True
                self.source_status.content = "源动作：BVH 骨架"
                self.render()
        elif name == "soma_ready":
            item, mesh = value
            if self.loaded_item and item["review_id"] == self.loaded_item["review_id"] and item.get("final_sha256") == self.loaded_item.get("final_sha256"):
                if self.soma_body.value:
                    try:
                        self.scene.set_source_mesh(mesh)
                        self.source_opacity.disabled = False
                        self.source_status.content = "源动作：SOMA 人体（已缓存）"
                        self.render()
                    except (ValueError, RuntimeError) as exc:
                        self.event("soma_error", (item, str(exc)))
        elif name == "soma_error":
            item, error = value
            if self.loaded_item and item["review_id"] == self.loaded_item["review_id"] and self.soma_body.value:
                self.source_status.content = f"SOMA 人体准备失败，骨架仍可审核：{error}"
        elif name == "playback":
            if value in {"播放", "暂停"} and self.clock and not self.loading:
                if not self.playing and int(self.clock.position) == self.clock.frames - 1:
                    self.clock.seek(0)
                    self.render()
                self.set_playing(not self.playing)
            elif value.startswith("循环"):
                self.looping = not self.looping
                self.update_playback_buttons()
        elif name == "save_review":
            try:
                self.save_current_review(value)
            except (ValueError, OSError, KeyError) as exc:
                self.review_status.content = f"审核记录未保存：{exc}"
        elif name == "refresh_batch":
            self.reload_batch()
        elif name == "pipeline":
            self.pipeline_action(value)
        elif name == "pipeline_progress":
            self.pipeline_note.content = value
        elif name in {"pipeline_result", "pipeline_error"}:
            self.loading = False
            self.refresh_pipeline()
            if name == "pipeline_error":
                self.pipeline_note.content += f"  \n操作未完成：{value}"
            elif value[0] in {"recommend", "next_cycle"} and value[1].get("items"):
                self.reload_batch()
            elif value[0] == "agent_review":
                self.pipeline_note.content += "  \nAI 对照报告已保存并通过格式与证据核验；待复核问题将优先进入混合人审。"
            elif value[0] == "recommend":
                self.pipeline_note.content += "  \n本轮发现已趋于稳定，可生成需要重处理和暂存的清单。"
            else:
                self.pipeline_note.content += "  \n已保存本轮重处理计划；待新产物注册后进入 AI 前后对照检查。"
        elif name == "loaded":
            try:
                self.install(*value)
            except Exception as exc:
                self.event("load_error", str(exc))
        elif name == "load_error":
            self.loading = False
            self.load.disabled = self.previous.disabled = self.next.disabled = False
            current = f'  \n当前画面仍为：{self.loaded_item["motion_id"]}' if self.scene and self.loaded_item else ""
            self.status.content = f"动作加载失败：{value}{current}"
            print(json.dumps({"event": "load_error", "error": value}, ensure_ascii=False), flush=True)
        elif name == "filter":
            self.filter()
        elif name == "load":
            self.start_loading(self.selected())
        elif name == "adjacent" and self.visible_items and not self.loading:
            selected = self.selected()
            index = self.visible_items.index(selected) if selected is not None else 0
            item = self.visible_items[(index + value) % len(self.visible_items)]
            self.motion.value = self.label(item)
            self.start_loading(item)
        elif name in {"seek", "step"} and self.clock:
            self.set_playing(False)
            frame = int(self.clock.position) + value if name == "step" else self.clock.frames - 1 if value == -1 else value
            self.clock.seek(frame)
            self.render()
        elif name == "render":
            self.render()
        elif name == "camera":
            self.camera()
        elif name == "client":
            self.camera(value)

    def run(self):
        last_tick = last_status = time.monotonic()
        try:
            while True:
                while not self.events.empty():
                    self.event(*self.events.get())
                now = time.monotonic()
                # Loading a scene or handling a resume click must not consume
                # motion time before playback actually starts.
                elapsed, last_tick = now - max(last_tick, getattr(self, "playback_started_at", last_tick)), now
                if self.clock and not self.loading and self.playing and self.server.get_clients():
                    previous = int(self.clock.position)
                    index, ended = self.clock.advance(elapsed, float(self.speed.value[:-1]), self.looping)
                    if index != previous:
                        self.render()
                    if ended:
                        self.set_playing(False)
                if now - last_status > .5:
                    self.update_preload_status()
                    last_status = now
                time.sleep(.005)
        except KeyboardInterrupt:
            pass
        finally:
            self.closed = True
            self.replay_buffer.close()
            self.soma_buffer.close()
            if self.scene:
                self.scene.close()
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run", type=Path)
    selection.add_argument("--session", type=Path, help="人工与AI迭代审核会话")
    selection.add_argument("--review-csv", type=Path, help="按处理复核CSV的完整原顺序预览动作")
    parser.add_argument("--motion-id", help="审核ID、动作名称或原始member，默认第一条")
    parser.add_argument("--motion-dir", type=Path, help="直接观察一个完整动作目录")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--prepare-only", action="store_true", help="只准备并验证所选动作，不启动网页")
    parser.add_argument("--report", type=Path, help="将验证摘要保存为新的JSON文件")
    args = parser.parse_args()
    try:
        if args.motion_dir and (args.run or args.session or args.review_csv):
            raise ValueError("直接动作目录与审核清单／会话请任选一种")
        if args.motion_dir:
            motion = args.motion_dir.resolve()
            items = [{"review_id": motion.name, "motion_id": motion.name, "member": motion.name,
                      "trajectory": str(motion / "trajectory.npz"), "sampling_roles": "direct"}]
        elif args.review_csv:
            items = csv_items(args.review_csv)
        elif args.session:
            items = adaptive_module().load_batch(args.session)
        else:
            if not args.run:
                raise ValueError("请指定 --motion-dir、--review-csv、--session 或 --run")
            items = sample_items(args.run)
        item = choose_item(items, args.motion_id)
        if args.prepare_only:
            result = replay_summary(prepare(item, args.cache), item)
            if args.report:
                with args.report.open("x") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return
        viewer = ReviewViewer(args, items)
        viewer.run()
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"replay: {exc}\n")


if __name__ == "__main__":
    main()
