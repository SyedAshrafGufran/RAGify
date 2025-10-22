# app_gui_final_v4.py
import sys
import os
import textwrap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox,
    QFileDialog, QCheckBox, QMenuBar, QAction, QFrame, QSplitter
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
        self.setGeometry(200, 100, 950, 780)
        self.dark_mode = False
        self.folder_path = None
        self.chat_history = []

        self.init_ui()
        self.show()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # === Menu Bar ===
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")

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
        header.setFont(QFont("Segoe UI", 26, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # === Folder Selection Bar ===
        folder_layout = QHBoxLayout()
        folder_label = QLabel("📁 Select Folder:")
        folder_label.setFont(QFont("Segoe UI", 11))

        self.folder_display = QLineEdit()
        self.folder_display.setReadOnly(True)
        self.folder_display.setPlaceholderText("No folder selected...")
        self.folder_display.setFont(QFont("Segoe UI", 11))
        self.folder_display.setStyleSheet("padding: 6px; border-radius: 5px;")

        select_button = QPushButton("Browse")
        select_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        select_button.setStyleSheet(
            "background-color: #1976D2; color: white; padding: 6px 12px; border-radius: 5px;"
        )
        select_button.clicked.connect(self.select_folder)

        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_display, stretch=1)
        folder_layout.addWidget(select_button)
        main_layout.addLayout(folder_layout)

        # === Splitter for Logs & Chat ===
        splitter = QSplitter(Qt.Vertical)

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
            "background-color: #f4f4f4; color: #222; border-radius: 8px; padding: 8px;"
        )
        log_layout.addWidget(self.log_display)
        log_frame.setLayout(log_layout)
        splitter.addWidget(log_frame)

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

        splitter.setSizes([250, 500])  # Log smaller, chat bigger
        main_layout.addWidget(splitter, stretch=1)

        # === Input Section ===
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your question here...")
        self.input_box.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(self.input_box, stretch=4)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.send_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 8px; border-radius: 6px;"
        )
        self.send_button.clicked.connect(self.handle_query)
        input_layout.addWidget(self.send_button, stretch=1)

        main_layout.addLayout(input_layout)

        # === Dark Mode Toggle ===
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setFont(QFont("Segoe UI", 10))
        self.dark_mode_toggle.stateChanged.connect(self.toggle_dark_mode)
        main_layout.addWidget(self.dark_mode_toggle, alignment=Qt.AlignRight)

        self.setLayout(main_layout)
        
    def format_answer_html(self, query, output, sources_with_chunks):
        # Escape any HTML special chars
        output_html = output.replace("\n", "<br>")
        sources_html = "<ul>" + "".join(f"<li>{s}</li>" for s in sources_with_chunks.split("\n")) + "</ul>"

        formatted_html = f"""
        <div style="font-family:Segoe UI; font-size:11pt;">
            <p><b>🧠 Answer for:</b> {query}</p>
            <hr style="border:1px solid #999;">
            <p>{output_html}</p>
            <p><b>📚 Sources used:</b></p>
            {sources_html}
            <hr style="border:1px solid #999;">
        </div>
        """
        return formatted_html

    # === Folder Selection ===
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Documents", os.getcwd()
        )
        if not folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder.")
            return

        self.folder_path = folder_path
        self.folder_display.setText(folder_path)
        self.log("📁 Selected folder: " + folder_path)
        self.log("🔍 Indexing documents...\n")

        def print_progress(msg):
            self.log(msg)

        try:
            index(folder_path, log_fn=print_progress)
            self.log("✅ Indexing completed!\n" + "-"*70)
        except Exception as e:
            self.log(f"❌ Indexing failed: {str(e)}")

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
            answer, sources_with_chunks = answer_query(query)  # Ensure you also get sources
        except Exception as e:
            answer = f"Error: {str(e)}"
            sources_with_chunks = ""

        html_answer = self.format_answer_html(query, answer, sources_with_chunks)
        self.append_chat(html_answer, user=False, html=True)
        self.chat_history.append(f"Bot: {answer}")

    # === Append Chat ===
    def append_chat(self, message, user=True, html=False):
        if html:
            self.chat_display.append(message)
        else:
            color = "#1E90FF" if user else "#32CD32"
            self.chat_display.setTextColor(QColor(color))
            self.chat_display.append(message)

        self.chat_display.append("<hr>")  # separator
        self.chat_display.moveCursor(self.chat_display.textCursor().End)
        self.chat_display.ensureCursorVisible()


    # === Append Logs ===
    def log(self, message):
        self.log_display.setTextColor(QColor("#8B0000"))
        self.log_display.append(message)
        self.log_display.moveCursor(self.log_display.textCursor().End)
        self.log_display.ensureCursorVisible()

    # === Save Conversation ===
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

    # === Help ===
    def show_help(self):
        QMessageBox.information(
            self,
            "Help",
            "1. Select a folder using the 'Browse' button.\n"
            "2. Wait for the logs to show indexing completion.\n"
            "3. Type a question below and click Send.\n"
            "4. Use File → Download Conversation to save chat.\n"
            "5. Toggle Dark Mode for comfort."
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
                "background-color: #f4f4f4; color: #222; border-radius: 8px; padding: 10px;"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotApp()
    sys.exit(app.exec_())
