# app_gui_final.py
import sys
import os
import textwrap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox,
    QFileDialog, QCheckBox, QFrame, QMenuBar, QAction
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

# Import your chatbot query function
from query import answer_query 
from index_docs import index 


class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚙️ RAGit")
        self.setGeometry(150, 100, 950, 700)
        self.dark_mode = False
        self.folder_path = None
        self.chat_history = []  # Store conversation for download
        self.init_ui()
        self.show()
        self.select_folder_after_open()

    def select_folder_after_open(self):
        """Ask for folder after the UI is shown."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Documents", os.getcwd()
        )
        if not folder_path:
            QMessageBox.critical(self, "No Folder Selected", "Application will exit.")
            sys.exit()
        else:
            self.folder_path = folder_path
            index(folder_path)
            self.append_chat(f"📁 Loaded folder: {folder_path}\n{'-'*70}\n", user=False)
            # Optional: create embeddings here using the selected folder
            # create_embeddings(folder_path)

    def init_ui(self):
        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Menu bar
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

        # Header
        header = QLabel("⚙️ RAGit")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Dark mode toggle
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setFont(QFont("Segoe UI", 11))
        self.dark_mode_toggle.stateChanged.connect(self.toggle_dark_mode)
        main_layout.addWidget(self.dark_mode_toggle, alignment=Qt.AlignRight)

        # Chat frame
        chat_frame = QFrame()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(5, 5, 5, 5)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        self.chat_display.setStyleSheet(
            "background-color: #f5f5f5; color: #1a1a1a; border-radius: 8px; padding: 10px;"
        )
        chat_layout.addWidget(self.chat_display)
        chat_frame.setLayout(chat_layout)
        main_layout.addWidget(chat_frame, stretch=1)

        # Input layout
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

    def toggle_dark_mode(self, state):
        self.dark_mode = state == Qt.Checked
        if self.dark_mode:
            self.setStyleSheet("background-color: #1e1e1e; color: #f0f0f0;")
            self.chat_display.setStyleSheet(
                "background-color: #2e2e2e; color: #e0e0e0; border-radius: 8px; padding: 10px;"
            )
        else:
            self.setStyleSheet("")
            self.chat_display.setStyleSheet(
                "background-color: #f5f5f5; color: #1a1a1a; border-radius: 8px; padding: 10px;"
            )

    def handle_query(self):
        query = self.input_box.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Input", "Please type a question!")
            return

        self.append_chat(f"User: {query}", user=True)
        self.chat_history.append(f"User: {query}")
        self.input_box.clear()

        try:
            answer = answer_query(query)  # Must return string
        except Exception as e:
            answer = f"Error: {str(e)}"

        wrapped_answer = textwrap.fill(answer, width=90)
        self.append_chat(f"Bot: {wrapped_answer}", user=False)
        self.chat_history.append(f"Bot: {wrapped_answer}")

    def append_chat(self, message, user=True):
        self.chat_display.setTextColor(QColor("#1E90FF") if user else QColor("#32CD32"))
        self.chat_display.append(message)
        self.chat_display.append("-" * 80 + "\n")
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

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

    def show_help(self):
        QMessageBox.information(
            self,
            "Help",
            "1. Select a folder with documents at startup.\n"
            "2. Type a question in the input box.\n"
            "3. Press Send to get answers.\n"
            "4. Use File > Download Conversation to save chat.\n"
            "5. Toggle Dark Mode using the checkbox."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotApp()
    sys.exit(app.exec_())
