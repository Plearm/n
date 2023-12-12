//#include <iostream>
//#include <mpi.h>
//#include <vector>
//#include <cstdlib>
//#include <ctime>
//
//int main(int argc, char** argv) {
//    setlocale(LC_ALL, "Russian");
//    MPI_Init(&argc, &argv);
//
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    if (world_size != 2) {
//        std::cerr << "This program is designed to run with 2 processes." << std::endl;
//        MPI_Abort(MPI_COMM_WORLD, 1);
//    }
//
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    const int numbers_count = 5;  // Количество случайных чисел для генерации
//
//    if (world_rank == 0) {
//        // Процесс 0 генерирует случайные числа и отправляет их процессу 1
//        std::vector<int> numbers(numbers_count);
//
//        // Генерация случайных чисел
//        srand(static_cast<unsigned>(time(nullptr)));
//        for (int i = 0; i < numbers_count; ++i) {
//            numbers[i] = rand() % 100;  // Генерация случайных чисел от 0 до 99
//        }
//
//        // Отправка чисел процессу 1
//        MPI_Send(numbers.data(), numbers_count, MPI_INT, 1, 0, MPI_COMM_WORLD);
//
//        // Получение результата от процесса 1
//        int sum;
//        MPI_Recv(&sum, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        // Вывод результата
//        std::cout << "Процесс 0: Получена сумма от процесса 1: " << sum << std::endl;
//    }
//    else if (world_rank == 1) {
//        // Процесс 1 получает числа от процесса 0, суммирует их и отправляет результат обратно
//        std::vector<int> received_numbers(numbers_count);
//
//        // Получение чисел от процесса 0
//        MPI_Recv(received_numbers.data(), numbers_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        // Вычисление суммы
//        int sum = 0;
//        for (int num : received_numbers) {
//            sum += num;
//        }
//
//        // Отправка результата процессу 0
//        MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//    }
//
//    MPI_Finalize();
//
//    return 0;
//}










				//Синхронное


//#include <iostream>
//#include <mpi.h>
//#include <cstdlib>
//#include <ctime>
//using namespace std;
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//    setlocale(LC_ALL, "Russian");
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    if (world_size != 2) {
//        std::cerr << "This program is designed to run with 2 processes." << std::endl;
//        //MPI_Abort(MPI_COMM_WORLD, 1);
//        MPI_Finalize();
//        return 0;
//    }
//
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    const int numbers_count = 5;  // Количество случайных чисел для генерации
//
//    if (world_rank == 0) {
//        // Процесс 0 генерирует случайные числа и отправляет их процессу 1
//        srand(static_cast<unsigned>(time(nullptr)));
//        int* numbers = new int[numbers_count];
//
//        // Генерация случайных чисел
//        for (int i = 0; i < numbers_count; ++i) {
//            numbers[i] = rand() % 100;  // Генерация случайных чисел от 0 до 99
//            cout << numbers[i] << endl;
//        }
//
//        // Отправка чисел процессу 1
//        MPI_Send(numbers, numbers_count, MPI_INT, 1, 0, MPI_COMM_WORLD);
//
//        // Получение результата от процесса 1
//        int sum;
//        MPI_Recv(&sum, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        // Вывод результата
//        cout << "Процесс 0: Получена сумма от процесса 1: " << sum << endl;
//
//        delete[] numbers;
//    }
//    else if (world_rank == 1) {
//        // Процесс 1 получает числа от процесса 0, суммирует их и отправляет результат обратно
//        int* received_numbers = new int[numbers_count];
//
//        // Получение чисел от процесса 0
//        MPI_Recv(received_numbers, numbers_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        // Вычисление суммы
//        int sum = 0;
//        for (int i = 0; i < numbers_count; ++i) {
//            sum += received_numbers[i];
//        }
//
//        if (world_rank == 0) {
//            // Вывод результата на экран только из процесса 0
//            std::cout << "Процесс 0: Получена сумма от процесса 1: " << sum << std::endl;
//        }
//
//        // Отправка результата процессу 0
//        MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//
//        delete[] received_numbers;
//    }
//
//    MPI_Finalize();
//
//    return 0;
//}





				//Асинхронное


#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
using namespace std;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация MPI

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Получение общего числа процессов

    if (world_size != 2) {
        std::cerr << "This program is designed to run with 2 processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // Аварийное завершение программы, если число процессов не равно 2
    }

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Получение номера текущего процесса

    const int numbers_count = 5;  // Количество случайных чисел для генерации

    if (world_rank == 0) {
        // Процесс 0 генерирует случайные числа и отправляет их процессу 1
        srand(static_cast<unsigned>(time(nullptr)));
        int* numbers = new int[numbers_count]; // Выделение динамической памяти под массив чисел

        // Генерация случайных чисел
        for (int i = 0; i < numbers_count; ++i) {
            numbers[i] = rand() % 100;  // Генерация случайных чисел от 0 до 99
        }

        MPI_Request send_request;
        // Асинхронная отправка чисел процессу 1
        MPI_Isend(numbers, numbers_count, MPI_INT, 1, 0, MPI_COMM_WORLD, &send_request);

        // Дополнительный код, который может выполняться параллельно с отправкой

        MPI_Wait(&send_request, MPI_STATUS_IGNORE); // Ожидание завершения асинхронной отправки

        delete[] numbers; // Освобождение динамической памяти
    }
    else if (world_rank == 1) {
        // Процесс 1 получает числа от процесса 0, суммирует их и отправляет результат обратно
        int* received_numbers = new int[numbers_count]; // Выделение динамической памяти под массив чисел

        MPI_Request recv_request;
        // Асинхронное получение чисел от процесса 0
        MPI_Irecv(received_numbers, numbers_count, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_request);

        // Дополнительный код, который может выполняться параллельно с получением

        MPI_Wait(&recv_request, MPI_STATUS_IGNORE); // Ожидание завершения асинхронного приема

        // Вычисление суммы
        int sum = 0;
        for (int i = 0; i < numbers_count; ++i) {
            sum += received_numbers[i];
        }
        
        cout << "Процесс 0: Получена сумма от процесса 1: " << sum << endl;
        
        delete[] received_numbers; // Освобождение динамической памяти

        // Отправка результата процессу 0
        MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }


    MPI_Finalize(); // Завершение работы с MPI

    return 0;
}






            //Коллективное


//#include <iostream>
//#include <mpi.h>
//#include <cstdlib>
//#include <ctime>
//
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv); // Инициализация MPI
//
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Получение общего числа процессов
//
//    if (world_size != 2) {
//        std::cerr << "This program is designed to run with 2 processes." << std::endl;
//        MPI_Abort(MPI_COMM_WORLD, 1); // Аварийное завершение программы, если число процессов не равно 2
//    }
//
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Получение номера текущего процесса
//
//    const int numbers_count = 5;  // Количество случайных чисел для генерации
//
//    int* numbers = new int[numbers_count]; // Выделение динамической памяти под массив чисел
//
//    // Генерация случайных чисел
//    srand(static_cast<unsigned>(time(nullptr)));
//    for (int i = 0; i < numbers_count; ++i) {
//        numbers[i] = rand() % 100;  // Генерация случайных чисел от 0 до 99
//    }
//
//    // Вычисление суммы чисел
//    int sum = 0;
//    for (int i = 0; i < numbers_count; ++i) {
//        sum += numbers[i];
//    }
//
//    int global_sum;
//
//    // Суммирование результатов всех процессов и сохранение в переменной global_sum
//    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//
//    if (world_rank == 0) {
//        // Вывод результата на экран только из процесса 0
//        std::cout << "Процесс 0: Получена сумма от процесса 1: " << global_sum << std::endl;
//    }
//
//    delete[] numbers; // Освобождение динамической памяти
//
//    MPI_Finalize(); // Завершение работы с MPI
//
//    return 0;
//}

//Здесь используется MPI_Reduce для суммирования результатов всех процессов и 
// получения общей суммы в процессе с рангом 0. Однако, в данном контексте это 
// может быть избыточным и менее эффективным, чем прямой обмен сообщениями.


