# app_gui_final_v3.py
import sys
import os
import textwrap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox,
    QFileDialog, QCheckBox, QFrame, QMenuBar, QAction, QSplitter
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

# Import chatbot logic
from query import answer_query
from index_docs import index


class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚙️ RAGit")
        self.setGeometry(150, 100, 950, 750)
        self.dark_mode = False
        self.folder_path = None
        self.chat_history = []

        self.init_ui()
        self.show()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # === Menu Bar ===
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")

        select_folder_action = QAction("Select Folder", self)
        select_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(select_folder_action)

        download_action = QAction("Download Conversation", self)
        download_action.triggered.connect(self.download_conversation)
        file_menu.addAction(download_action)

        help_menu = menu_bar.addMenu("Help")
        help_action = QAction("Instructions", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        main_layout.setMenuBar(menu_bar)

        # === Header ===
        header = QLabel("⚙️ RAGit")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # === Dark Mode Toggle ===
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setFont(QFont("Segoe UI", 11))
        self.dark_mode_toggle.stateChanged.connect(self.toggle_dark_mode)
        main_layout.addWidget(self.dark_mode_toggle, alignment=Qt.AlignRight)

        # === Splitter for Chat & Logs ===
        splitter = QSplitter(Qt.Vertical)

        # --- Chat Section ---
        chat_frame = QFrame()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(5, 5, 5, 5)

        chat_label = QLabel("💬 Chat")
        chat_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        chat_layout.addWidget(chat_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        self.chat_display.setStyleSheet(
            "background-color: #f8f8f8; color: #111; border-radius: 8px; padding: 10px;"
        )
        chat_layout.addWidget(self.chat_display)
        chat_frame.setLayout(chat_layout)
        splitter.addWidget(chat_frame)

        # --- Log Section ---
        log_frame = QFrame()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)

        log_label = QLabel("🧾 Logs")
        log_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        log_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet(
            "background-color: #f0f0f0; color: #333; border-radius: 8px; padding: 10px;"
        )
        log_layout.addWidget(self.log_display)
        log_frame.setLayout(log_layout)
        splitter.addWidget(log_frame)

        splitter.setSizes([500, 200])  # Adjust proportions
        main_layout.addWidget(splitter, stretch=1)

        # === Input Area ===
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your question here...")
        self.input_box.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(self.input_box, stretch=4)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Segoe UI", 12))
        self.send_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 8px; border-radius: 6px;"
        )
        self.send_button.clicked.connect(self.handle_query)
        input_layout.addWidget(self.send_button, stretch=1)

        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

    # === Folder Selection & Indexing ===
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Documents", os.getcwd()
        )
        if not folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder.")
            return

        self.folder_path = folder_path
        self.log("📁 Selected folder: " + folder_path)
        self.log("🔍 Indexing documents...\n")

        def print_progress(msg):
            self.log(msg)

        try:
            index(folder_path, log_fn=print_progress)
            self.log("✅ Indexing completed!\n" + "-"*70)
        except Exception as e:
            self.log(f"❌ Indexing failed: {str(e)}")

    # === Chat Handling ===
    def handle_query(self):
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first!")
            return

        query = self.input_box.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Input", "Please type a question!")
            return

        self.append_chat(f"🧑 You: {query}", user=True)
        self.chat_history.append(f"You: {query}")
        self.input_box.clear()

        try:
            answer = answer_query(query)
        except Exception as e:
            answer = f"Error: {str(e)}"

        wrapped = textwrap.fill(answer, width=90)
        self.append_chat(f"🤖 Bot: {wrapped}", user=False)
        self.chat_history.append(f"Bot: {wrapped}")

    # === Append to Chat ===
    def append_chat(self, message, user=True):
        color = "#1E90FF" if user else "#32CD32"
        self.chat_display.setTextColor(QColor(color))
        self.chat_display.append(message)
        self.chat_display.append("-" * 80 + "\n")
        self.chat_display.moveCursor(self.chat_display.textCursor().End)
        self.chat_display.ensureCursorVisible()

    # === Append to Logs ===
    def log(self, message):
        self.log_display.setTextColor(QColor("#8B0000"))
        self.log_display.append(message)
        self.log_display.moveCursor(self.log_display.textCursor().End)
        self.log_display.ensureCursorVisible()

    # === Conversation Save ===
    def download_conversation(self):
        if not self.chat_history:
            QMessageBox.information(self, "No Conversation", "There is nothing to save yet.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Conversation", os.getcwd(), "Text Files (*.txt)"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chat_history))
            QMessageBox.information(self, "Saved", f"Conversation saved at {file_path}")

    # === Help Dialog ===
    def show_help(self):
        QMessageBox.information(
            self,
            "Help",
            "1. Use File → Select Folder to load your documents.\n"
            "2. Type a question below and press Send.\n"
            "3. Logs show progress and indexing details.\n"
            "4. Save your chat via File → Download Conversation.\n"
            "5. Use Dark Mode toggle for comfort."
        )

    # === Dark Mode ===
    def toggle_dark_mode(self, state):
        self.dark_mode = state == Qt.Checked
        if self.dark_mode:
            self.setStyleSheet("background-color: #121212; color: #f0f0f0;")
            self.chat_display.setStyleSheet(
                "background-color: #1e1e1e; color: #e0e0e0; border-radius: 8px; padding: 10px;"
            )
            self.log_display.setStyleSheet(
                "background-color: #242424; color: #c0c0c0; border-radius: 8px; padding: 10px;"
            )
        else:
            self.setStyleSheet("")
            self.chat_display.setStyleSheet(
                "background-color: #f8f8f8; color: #111; border-radius: 8px; padding: 10px;"
            )
            self.log_display.setStyleSheet(
                "background-color: #f0f0f0; color: #333; border-radius: 8px; padding: 10px;"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotApp()
    sys.exit(app.exec_())
