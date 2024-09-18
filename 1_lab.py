import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Menu
from PIL import Image, ImageTk
import numpy as np

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторна робота №1(Просторова фільтрація зображень)")
        self.image = None
        self.original_image = None  # Оригінальне зображення

        # Додаємо новий Label для оригінального зображення
        self.original_img_label = tk.Label(root)
        self.original_img_label.grid(row=0, column=0, columnspan=5)  # Оригінальне зображення

        # Існуючий Label для зображення після фільтру
        self.img_label = tk.Label(root)
        self.img_label.grid(row=0, column=5, columnspan=5)  # Оброблене зображення

        # Додаємо мітку для відображення ядра матриці
        self.kernel_label = tk.Label(root, text="Ядро фільтру:")
        self.kernel_label.grid(row=2, column=0, columnspan=10)

        self.create_menu()

    def create_menu(self):
        # Створюємо меню
        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Завантажити зображення", command=self.load_image)
        file_menu.add_command(label="Зберегти зображення", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Вийти", command=self.root.quit)

        filter_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Фільтри", menu=filter_menu)
        filter_menu.add_command(label="Гаус 3x3", command=lambda: self.apply_filter('gaussian_3x3'))
        filter_menu.add_command(label="Гаус 5x5", command=lambda: self.apply_filter('gaussian_5x5'))
        filter_menu.add_command(label="Лаплас 3x3", command=lambda: self.apply_filter('laplacian_3x3'))
        filter_menu.add_command(label="Лаплас 5x5", command=lambda: self.apply_filter('laplacian_5x5'))
        filter_menu.add_command(label="Пріюта вертик.", command=lambda: self.apply_filter('prewitt_vertical'))
        filter_menu.add_command(label="Пріюта гориз.", command=lambda: self.apply_filter('prewitt_horizontal'))
        filter_menu.add_command(label="Собель вертик.", command=lambda: self.apply_filter('sobel_vertical'))
        filter_menu.add_command(label="Собель гориз.", command=lambda: self.apply_filter('sobel_horizontal'))
        filter_menu.add_command(label="Hipass", command=lambda: self.apply_filter('hipass'))
        filter_menu.add_command(label="Sharpen", command=lambda: self.apply_filter('sharpen'))
        filter_menu.add_command(label="Пом'якшення", command=lambda: self.apply_filter('blur'))
        filter_menu.add_separator()
        filter_menu.add_command(label="Користувацький фільтр", command=self.custom_filter)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if file_path:
            self.image = Image.open(file_path)
            self.original_image = self.image.copy()  # Збереження оригіналу
            self.display_image(self.original_image, is_original=True)

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                self.image.save(file_path)
        else:
            messagebox.showerror("Помилка", "Немає зображення для збереження")

    def display_image(self, image, is_original=False):
        img_display = ImageTk.PhotoImage(image)
        if is_original:
            self.original_img_label.config(image=img_display)
            self.original_img_label.image = img_display  # Оригінальне зображення
        else:
            self.img_label.config(image=img_display)
            self.img_label.image = img_display  # Оброблене зображення

    def apply_filter(self, filter_type):
        if self.image:
            img_array = np.array(self.original_image.convert('L'), dtype=np.float32)
            
            kernels = {
                'gaussian_3x3': np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]], dtype=np.float32),
                'gaussian_5x5': np.array([[2, 7, 12, 7, 2],
                                          [7, 31, 52, 31, 7],
                                          [12, 52, 127, 52, 12],
                                          [7, 31, 52, 31, 7],
                                          [2, 7, 12, 7, 2]], dtype=np.float32),
                'laplacian_3x3': np.array([[-1, -1, -1],
                                           [-1,  8, -1],
                                           [-1, -1, -1]], dtype=np.float32),
                'laplacian_5x5': np.array([[-1, -3, -4, -3, -1],
                                           [-3,  0,  6,  0, -3],
                                           [-4,  6, 20,  6, -4],
                                           [-3,  0,  6,  0, -3],
                                           [-1, -3, -4, -3, -1]], dtype=np.float32),
                'prewitt_vertical': np.array([[-1,  0,  1],
                                              [-1,  0,  1],
                                              [-1,  0,  1]], dtype=np.float32),
                'prewitt_horizontal': np.array([[ 1,  1,  1],
                                                [ 0,  0,  0],
                                                [-1, -1, -1]], dtype=np.float32),
                'sobel_vertical': np.array([[-1,  0,  1],
                                            [-2,  0,  2],
                                            [-1,  0,  1]], dtype=np.float32),
                'sobel_horizontal': np.array([[ 1,  2,  1],
                                              [ 0,  0,  0],
                                              [-1, -2, -1]], dtype=np.float32),
                'hipass': np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=np.float32),
                'sharpen': np.array([[ 0, -1,  0],
                                     [-1,  5, -1],
                                     [ 0, -1,  0]], dtype=np.float32),
                'blur': np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]], dtype=np.float32) / 9
            }
            
            kernel = kernels.get(filter_type)
            if kernel is None:
                messagebox.showerror("Помилка", "Невідомий тип фільтра")
                return

            # Display the kernel before normalization
            kernel_str = '\n'.join([' '.join(f"{round(num)}" for num in row) for row in kernel])
            self.kernel_label.config(text=f"Ядро фільтру:\n{kernel_str}")

            # Normalize kernel
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                kernel = kernel / kernel_sum

            # Convolution operation
            kernel_height, kernel_width = kernel.shape
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2

            # Pad the image
            padded_image = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

            # Create an empty array for the output image
            filtered_img = np.zeros_like(img_array)

            # Apply the kernel to each pixel
            for y in range(img_array.shape[0]):
                for x in range(img_array.shape[1]):
                    # Extract the region of interest
                    region = padded_image[y:y+kernel_height, x:x+kernel_width]
                    # Apply the kernel
                    filtered_value = np.sum(region * kernel)
                    # Store the result
                    filtered_img[y, x] = filtered_value

            # Clamp pixel values to the range [0, 255]
            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            # Convert to image and display
            self.image = Image.fromarray(filtered_img)
            self.display_image(self.image)
        else:
            messagebox.showerror("Помилка", "Завантажте зображення")

    def custom_filter(self):
        if self.image:
            # Введення розміру фільтра
            size = simpledialog.askinteger("Розмір фільтра", "Введіть розмір фільтра (наприклад, 3 або 5):", minvalue=3, maxvalue=15)
            if not size or size % 2 == 0:
                messagebox.showerror("Помилка", "Неправильний розмір фільтра")
                return

            # Створення матриці фільтра
            filter_matrix = []
            for i in range(size):
                try:
                    row_input = simpledialog.askstring("Коефіцієнти фільтра", f"Введіть коефіцієнти для рядка {i + 1} (через пробіл):")
                    filter_row = list(map(float, row_input.split()))
                    if len(filter_row) != size:
                        raise ValueError("Неправильна кількість коефіцієнтів")
                    filter_matrix.append(filter_row)
                except ValueError:
                    messagebox.showerror("Помилка", "Неправильний формат вводу")
                    return

            if len(filter_matrix) != size or any(len(row) != size for row in filter_matrix):
                messagebox.showerror("Помилка", "Матриця повинна бути квадратною")
                return

            kernel = np.array(filter_matrix, dtype=np.float32)

            self.apply_custom_filter(kernel)
        else:
            messagebox.showerror("Помилка", "Завантажте зображення")

    def apply_custom_filter(self, kernel):
        if self.image:
            # Convert image to grayscale and array format
            img_array = np.array(self.original_image.convert('L'), dtype=np.float32)

            # Display the custom kernel before normalization
            kernel_str = '\n'.join([' '.join(f"{round(num)}" for num in row) for row in kernel])
            self.kernel_label.config(text=f"Ядро фільтру:\n{kernel_str}")

            # Normalize kernel
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                kernel = kernel / kernel_sum

            # Convolution operation
            kernel_height, kernel_width = kernel.shape
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2

            # Pad the image
            padded_image = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

            # Create an empty array for the output image
            filtered_img = np.zeros_like(img_array)

            # Apply the kernel to each pixel
            for y in range(img_array.shape[0]):
                for x in range(img_array.shape[1]):
                    # Extract the region of interest
                    region = padded_image[y:y+kernel_height, x:x+kernel_width]
                    # Apply the kernel
                    filtered_value = np.sum(region * kernel)
                    # Store the result
                    filtered_img[y, x] = filtered_value

            # Clamp pixel values to the range [0, 255]
            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            # Convert to image and display
            self.image = Image.fromarray(filtered_img)
            self.display_image(self.image)
        else:
            messagebox.showerror("Помилка", "Завантажте зображення")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop() 