<?php
/**
 * Реализовать возможность входа с паролем и логином с использованием
 * сессии для изменения отправленных данных в предыдущей задаче,
 * пароль и логин генерируются автоматически при первоначальной отправке формы.
 */
function randomPassword() {
    $alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890';
    $pass = array();
    $alphaLength = strlen($alphabet) - 1;
    $ram = rand(15, 20);
    for ($i = 0; $i < $ram; $i++) {
        $n = rand(0, $alphaLength);
        $pass[] = $alphabet[$n];
    }
    return implode($pass); //turn the array into a string
}

// Отправляем браузеру правильную кодировку,
// файл index.php должен быть в кодировке UTF-8 без BOM.
header('Content-Type: text/html; charset=UTF-8');
$user = 'u52830';
$pass = '7841698';
$db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, [PDO::ATTR_PERSISTENT => true]);

// В суперглобальном массиве $_SERVER PHP сохраняет некторые заголовки запроса HTTP
// и другие сведения о клиненте и сервере, например метод текущего запроса $_SERVER['REQUEST_METHOD'].
if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    // Массив для временного хранения сообщений пользователю.
    $messages = array();

    // В суперглобальном массиве $_COOKIE PHP хранит все имена и значения куки текущего запроса.
    // Выдаем сообщение об успешном сохранении.
    if (!empty($_COOKIE['save'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('save', '', 100000);        
        // Выводим сообщение пользователю.
        $messages[] = 'Спасибо, результаты сохранены.';
        // Если в куках есть пароль, то выводим сообщение.
        if (!empty($_COOKIE['pass'])) {
            $messages[] = sprintf('Вы можете <a href="login.php">войти</a> с логином <strong>%s</strong>
        и паролем <strong>%s</strong> для изменения данных.',
                strip_tags($_COOKIE['login']),
                strip_tags($_COOKIE['pass']));
        }
        setcookie('login', '', 100000);
        setcookie('pass', '', 100000);
    }

    // Складываем признак ошибок в массив.
    $errors = array();
    $errors['fio'] = !empty($_COOKIE['fio_error']);
    $errors['email'] = !empty($_COOKIE['email_error']);
    $errors['year'] = !empty($_COOKIE['year_error']);
    $errors['limbs'] = !empty($_COOKIE['limbs_error']);
    $errors['pol'] = !empty($_COOKIE['pol_error']);
    $errors['super'] = !empty($_COOKIE['super_error']);
    $errors['biography'] = !empty($_COOKIE['biography_error']);
    $errors['check-1'] = !empty($_COOKIE['check_1_error']);
    // TODO: аналогично все поля.

    // Выдаем сообщения об ошибках.
    if (!empty($errors['fio'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('fio_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните имя.</div>';
    }
    if (!empty($errors['email'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('email_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните почту.</div>';
    }
    if (!empty($errors['year'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('year_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните год рождения.</div>';
    }
    if (!empty($errors['limbs'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('limbs_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните количество конечностей.</div>';
    }
    if (!empty($errors['pol'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('pol_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните пол.</div>';
    }
    if (!empty($errors['super'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('super_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните сверхспособности.</div>';
    }
    if (!empty($errors['biography'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('biography_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните биографию.</div>';
    }
    if (!empty($errors['check-1'])) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('check_1_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Поставьте галочку.</div>';
    }
    // TODO: тут выдать сообщения об ошибках в других полях.

    // Складываем предыдущие значения полей в массив, если есть.
    // При этом санитизуем все данные для безопасного отображения в браузере.
    $values = array();
    $values['fio'] = empty($_COOKIE['fio_value']) ? '' : htmlspecialchars(strip_tags($_COOKIE['fio_value']));
    $values['email'] = empty($_COOKIE['email_value']) ? '' : htmlspecialchars(strip_tags($_COOKIE['email_value']));
    $values['year'] = empty($_COOKIE['year_value']) ? '' : (int)$_COOKIE['year_value'];
    $values['limbs'] = empty($_COOKIE['limbs_value']) ? '' : (int)$_COOKIE['limbs_value'];
    $values['pol'] = empty($_COOKIE['pol_value']) ? '' : htmlspecialchars(strip_tags($_COOKIE['pol_value']));
    $values['super'] = empty($_COOKIE['super_value']) ? '' : unserialize($_COOKIE['super_value']);
    $values['biography'] = empty($_COOKIE['biography_value']) ? '' : htmlspecialchars(strip_tags($_COOKIE['biography_value']));
    $values['check-1'] = empty($_COOKIE['check_1_value']) ? '' : htmlspecialchars(strip_tags($_COOKIE['check_1_value']));
    // TODO: аналогично все поля.

    // Если нет предыдущих ошибок ввода, есть кука сессии, начали сессию и
    // ранее в сессию записан факт успешного логина.
    if (count(array_filter($errors)) === 0 && !empty($_COOKIE[session_name()]) &&
        session_start() && !empty($_SESSION['login'])) {
        $_SESSION['token'] = bin2hex(random_bytes(32));
        // TODO: загрузить данные пользователя из БД
        // и заполнить переменную $values,
        // предварительно санитизовав.
        $login = $_SESSION['login'];
        try {
            $stmt = $db->prepare("SELECT app_id FROM user WHERE login = ?");
            $stmt->execute([$login]);
            $app_id = $stmt->fetchColumn();

            $stmt = $db->prepare("SELECT name,email,year,pol,kol_kon,biography FROM application WHERE id = ?");
            $stmt->execute([$app_id]);
            $app = $stmt->fetchAll(PDO::FETCH_ASSOC);

            $stmt = $db->prepare("SELECT idsuper FROM userconnection WHERE idap = ?");
            $stmt->execute([$app_id]);
            $abilities = $stmt->fetchAll(PDO::FETCH_COLUMN, 0);

            if (!empty($app[0]['name'])) {
                $values['fio'] = htmlspecialchars(strip_tags($app[0]['name']));
            }
            if (!empty($app[0]['email'])) {
                $values['email'] = htmlspecialchars(strip_tags($app[0]['email']));
            }
            if (!empty($app[0]['year'])) {
                $values['year'] = $app[0]['year'];
            }
            if (!empty($app[0]['kol_kon'])) {
                $values['limbs'] = $app[0]['kol_kon'];
            }
            if (!empty($app[0]['pol'])) {
                $values['pol'] = $app[0]['pol'];
            }
            if (!empty($app[0]['biography'])) {
                $values['biography'] = htmlspecialchars(strip_tags($app[0]['biography']));
            }
            if (!empty($abilities)) {
                $values['super'] =  $abilities;
            }
            $values['check-1'] = 1;
        } catch (PDOException $e) {
            print('Error : ' . $e->getMessage());
            exit();
        }
        ///////

        printf('Вход с логином %s, uid %d', $_SESSION['login'], $_SESSION['uid']);
    }
    // Включаем содержимое файла form.php.
    // В нем будут доступны переменные $messages, $errors и $values для вывода
    // сообщений, полей с ранее заполненными данными и признаками ошибок.
    include('form.php');
}
// Иначе, если запрос был методом POST, т.е. нужно проверить данные и сохранить их в XML-файл.
else {
    // Проверяем ошибки.
    $errors = FALSE;
    if (empty($_POST['fio']) || !preg_match('/^([a-zA-Z\'\-]+\s*|[а-яА-ЯёЁ\'\-]+\s*)$/u', $_POST['fio'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('fio_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('fio_value', $_POST['fio'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['email']) || !preg_match('/^((([0-9A-Za-z]{1}[-0-9A-z\.]{1,}[0-9A-Za-z]{1})|([0-9А-Яа-я]{1}[-0-9А-я\.]{1,}[0-9А-Яа-я]{1}))@([-A-Za-z]{1,}\.){1,2}[-A-Za-z]{2,})$/u', $_POST['email'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('email_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('email_value', $_POST['email'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['year']) || !is_numeric($_POST['year']) || !preg_match('/^\d+$/', $_POST['year'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('year_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('year_value', $_POST['year'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['limbs']) || !is_numeric($_POST['limbs']) || ($_POST['limbs'] < 1) || ($_POST['limbs'] > 5)) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('limbs_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('limbs_value', $_POST['limbs'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['pol']) || !($_POST['pol'] == 'M' || $_POST['pol'] == 'W')) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('pol_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('pol_value', $_POST['pol'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['super']) || !is_array($_POST['super']) || (int)$_POST['super'] < 1 || (int)$_POST['super'] > 3) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('super_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('super_value', serialize($_POST['super']), time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['biography'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('biography_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('biography_value', $_POST['biography'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['check-1']) || !($_POST['check-1'] == 'on' || $_POST['check-1'] == 1)) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('check_1_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    } else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('check_1_value', $_POST['check-1'], time() + 30 * 24 * 60 * 60);
    }

    // *************
    // TODO: тут необходимо проверить правильность заполнения всех остальных полей.
    // Сохранить в Cookie признаки ошибок и значения полей.
    // *************

    if ($errors) {
        // При наличии ошибок перезагружаем страницу и завершаем работу скрипта.
        header('Location: index.php');
        exit();
    } else {
        // Удаляем Cookies с признаками ошибок.
        setcookie('fio_error', '', 100000);
        setcookie('email_error', '', 100000);
        setcookie('year_error', '', 100000);
        setcookie('limbs_error', '', 100000);
        setcookie('pol_error', '', 100000);
        setcookie('super_error', '', 100000);
        setcookie('biography_error', '', 100000);
        setcookie('check_1_error', '', 100000);
        // TODO: тут необходимо удалить остальные Cookies.
    }

    // Проверяем меняются ли ранее сохраненные данные или отправляются новые.
    if (!empty($_COOKIE[session_name()]) &&
        session_start() && !empty($_SESSION['login'])) {
        if (!empty($_POST['token']) && hash_equals($_POST['token'], $_SESSION['token'])) {
        //echo 'good';
        // TODO: перезаписать данные в БД новыми данными,
        // кроме логина и пароля.
        try {
            $stmt = $db->prepare("SELECT app_id FROM user WHERE login = ?");
            $stmt->execute([$_SESSION['login']]);
            $app_id = $stmt->fetchColumn();
            $stmt = $db->prepare("UPDATE application SET name = ?, email = ?, year = ?, pol = ?, kol_kon = ?, biography = ? WHERE id = ?");
            $stmt->execute([$_POST['fio'], $_POST['email'], (int)$_POST['year'], $_POST['pol'], (int)$_POST['limbs'], $_POST['biography'], $app_id]);
            $stmt = $db->prepare("SELECT idsuper FROM userconnection WHERE idap = ?");
            $stmt->execute([$app_id]);
            $abil = $stmt->fetchAll(PDO::FETCH_COLUMN, 0);
            var_dump($abil);
            var_dump($_POST['super']);
            if (array_diff($abil, $_POST['super']) || count($abil) != count($_POST['super'])) {
                $stmt = $db->prepare("DELETE FROM userconnection WHERE idap = ?");
                $stmt->execute([$app_id]);

                foreach ($_POST['super'] as $super_id) {
                    $stmt = $db->prepare("INSERT INTO userconnection SET idap = ?, idsuper = ?");
                    $stmt->execute([$app_id, $super_id]);
                }
            }

        } catch (PDOException $e) {
            print('Error : ' . $e->getMessage());
            exit();
        }
        } else {
            var_dump($_POST['token']);
            var_dump($_SESSION['token']);
            die('Ошибка CSRF: недопустимый токен');
        }
    } else {
        //echo 'gr';
        // Генерируем уникальный логин и пароль.
        // TODO: сделать механизм генерации, например функциями rand(), uniquid(), md5(), substr().
        $login = uniqid();
        $password = randomPassword();
        // Сохраняем в Cookies.
        setcookie('login', $login);
        setcookie('pass', $password);

        try {
            $stmt = $db->prepare("REPLACE INTO application SET name = ?,email = ?,year = ?,pol = ?,kol_kon = ?,biography = ?,ccheck = ?");
            $stmt->execute([$_POST['fio'], $_POST['email'], (int)$_POST['year'], $_POST['pol'], (int)$_POST['limbs'], $_POST['biography'], 1]);
        } catch (PDOException $e) {
            print('Error : ' . $e->getMessage());
            exit();
        }

        $app_id = $db->lastInsertId();

        foreach ($_POST['super'] as $super) {
            try {
                $stmt = $db->prepare("REPLACE INTO userconnection SET idap = ?, idsuper = ?");
                $stmt->execute([$app_id, $super]);
            } catch (PDOException $e) {
                print('Error : ' . $e->getMessage());
                exit();
            }
        }
        try {
            $stmt = $db->prepare("REPLACE INTO user SET app_id = ?, login = ?, password = ?");
            $stmt->execute([(int)$app_id, $login, md5($password)]);
        } catch (PDOException $e) {
            print('Error : ' . $e->getMessage());
            exit();
        }
        // TODO: Сохранение данных формы, логина и хеш md5() пароля в базу данных.
        // ...
    }
    // Сохраняем куку с признаком успешного сохранения.
        setcookie('save', '1');

//        Делаем перенаправление.
       header('Location: ./');
}
