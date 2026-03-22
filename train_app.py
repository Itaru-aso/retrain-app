
import customtkinter as ctk
from tkinter import messagebox
import threading
import sys
import io
import os
from pipline import TrainingPipeline
from omegaconf import OmegaConf

# 標準出力をUIに表示するためのリダイレクタ
class StdoutRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    def write(self, s):
        self.text_widget.insert("end", s)
        self.text_widget.see("end")
    def flush(self):
        pass

class TrainingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("学習パイプライン UI")
        self.geometry("900x600")
        ctk.set_appearance_mode("dark")  # "light" も選べます
        ctk.set_default_color_theme("blue")

        # --------- 左側：色番選択（複数） ----------
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")

        self.colors_label = ctk.CTkLabel(self.left_frame, text="色番を選択（複数可）")
        self.colors_label.pack(anchor="w", padx=10, pady=(10, 0))

        self.color_options = self.get_color_folders("./dataset")
        self.color_vars = {}  # {color_name: CTkCheckBox variable}

        self.scroll_colors = ctk.CTkScrollableFrame(self.left_frame, width=250, height=350)
        self.scroll_colors.pack(fill="both", expand=True, padx=10, pady=10)

        for name in self.color_options:
            var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(self.scroll_colors, text=name, variable=var)
            chk.pack(anchor="w", padx=8, pady=4)
            self.color_vars[name] = var

        # 全選択・解除ボタン
        btns_frame = ctk.CTkFrame(self.left_frame)
        btns_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.select_all_btn = ctk.CTkButton(btns_frame, text="すべて選択", command=self.select_all)
        self.clear_all_btn = ctk.CTkButton(btns_frame, text="選択解除", command=self.clear_all)
        self.select_all_btn.pack(side="left", padx=(0, 6))
        self.clear_all_btn.pack(side="left", padx=(6, 0))

        # 実行ボタン（複数対象）
        self.run_button = ctk.CTkButton(self.left_frame, text="選択した色番で実行", command=self.run_multiple)
        self.run_button.pack(fill="x", padx=10, pady=(0, 10))

        # --------- 右側：ログ表示 ----------
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.log_box = ctk.CTkTextbox(self.right_frame, width=600, height=520)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=20)

        # 標準出力をリダイレクト
        sys.stdout = StdoutRedirector(self.log_box)

    # フォルダ名取得関数
    def get_color_folders(self, base_path):
        try:
            return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        except FileNotFoundError:
            messagebox.showerror("フォルダエラー", f"{base_path} が見つかりません。")
            return []

    # 全選択
    def select_all(self):
        for var in self.color_vars.values():
            var.set(True)

    # 解除
    def clear_all(self):
        for var in self.color_vars.values():
            var.set(False)

    # 実行ボタン押下（複数色番）
    def run_multiple(self):
        selected = [name for name, var in self.color_vars.items() if var.get()]
        if not selected:
            messagebox.showinfo("選択なし", "少なくとも1つの色番を選択してください。")
            return

        # config の存在チェック（任意）
        if not os.path.exists("./conf/config.yaml"):
            messagebox.showerror("設定エラー", "./conf/config.yaml が見つかりません。")
            return

        self.run_button.configure(state="disabled")
        # 別スレッドで順次実行
        threading.Thread(target=self.execute_multiple, args=(selected,), daemon=True).start()

    # 複数色番を順次パイプライン実行
    def execute_multiple(self, color_list):
        try:
            total = len(color_list)
            for idx, color_value in enumerate(color_list, start=1):
                print(f"\n=== {idx}/{total} : 色番[{color_value}] の学習を開始 ===")
                cfg = OmegaConf.load("./conf/config.yaml")
                cfg.target_color = str(color_value)

                self.pipeline = TrainingPipeline(cfg)

                print("🚀 実行開始...\n")
                self.pipeline.execute()
                print("✅ 実行完了")

                """
                # 閾値読み込み（色番も表示）
                try:
                    cfg_m = OmegaConf.load("./conf/threshold_monochro.yaml")
                    cfg_c = OmegaConf.load("./conf/threshold_color.yaml")
                    mono = getattr(cfg_m, "threshold", None)
                    color = getattr(cfg_c, "threshold", None)
                    tc_m = getattr(cfg_m, "target_color", None)
                    tc_c = getattr(cfg_c, "target_color", None)
                    # target_color が閾値ファイルにも保存されるならそれを表示、無ければ現在の color_value を表示
                    shown_color = tc_m or tc_c or color_value
                    print(f"\n色番[{shown_color}] の閾値 ⇒ モノクロ: {mono}, カラー: {color}")
                except Exception as e_th:
                    print(f"\n⚠ 閾値読込エラー（色番[{color_value}]）: {e_th}")
                """

                print(f"=== {idx}/{total} : 色番[{color_value}] の学習終了 ===\n")
        except Exception as e:
            print(f"\n❌ 実行エラー: {e}")
        finally:
            # ボタンを必ず再有効化
            self.run_button.configure(state="normal")

if __name__ == "__main__":
    app = TrainingApp()
    app.mainloop()
