<?php
/**
 * Реализовать проверку заполнения обязательных полей формы в предыдущей
 * с использованием Cookies, а также заполнение формы по умолчанию ранее
 * введенными значениями.
 */

// Отправляем браузеру правильную кодировку,
// файл index.php должен быть в кодировке UTF-8 без BOM.
header('Content-Type: text/html; charset=UTF-8');

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
        // Если есть параметр save, то выводим сообщение пользователю.
        $messages[] = 'Спасибо, результаты сохранены.';
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
    if ($errors['fio']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('fio_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните имя.</div>';
    }
    if ($errors['email']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('email_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните почту.</div>';
    }
    if ($errors['year']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('year_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните год рождения.</div>';
    }
    if ($errors['limbs']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('limbs_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните количество конечностей.</div>';
    }
    if ($errors['pol']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('pol_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните пол.</div>';
    }
    if ($errors['super']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('super_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните сверхспособности.</div>';
    }
    if ($errors['biography']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('biography_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Заполните биографию.</div>';
    }
    if ($errors['check-1']) {
        // Удаляем куку, указывая время устаревания в прошлом.
        setcookie('check_1_error', '', 100000);
        // Выводим сообщение.
        $messages[] = '<div class="error">Поставьте галочку.</div>';
    }
    // TODO: тут выдать сообщения об ошибках в других полях.

    // Складываем предыдущие значения полей в массив, если есть.
    $values = array();
    $values['fio'] = empty($_COOKIE['fio_value']) ? '' : $_COOKIE['fio_value'];
    $values['email'] = empty($_COOKIE['email_value']) ? '' : $_COOKIE['email_value'];
    $values['year'] = empty($_COOKIE['year_value']) ? '' : (int)$_COOKIE['year_value'];
    $values['limbs'] = empty($_COOKIE['limbs_value']) ? '' : (int)$_COOKIE['limbs_value'];
    $values['pol'] = empty($_COOKIE['pol_value']) ? '' : $_COOKIE['pol_value'];
    $values['super'] = empty($_COOKIE['super_value']) ? '' : unserialize($_COOKIE['super_value']);
    $values['biography'] = empty($_COOKIE['biography_value']) ? '' : $_COOKIE['biography_value'];
    $values['check-1'] = empty($_COOKIE['check_1_value']) ? '' : $_COOKIE['check_1_value'];
    // TODO: аналогично все поля.

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
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('fio_value', $_POST['fio'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['email']) || !preg_match('/^((([0-9A-Za-z]{1}[-0-9A-z\.]{1,}[0-9A-Za-z]{1})|([0-9А-Яа-я]{1}[-0-9А-я\.]{1,}[0-9А-Яа-я]{1}))@([-A-Za-z]{1,}\.){1,2}[-A-Za-z]{2,})$/u', $_POST['email'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('email_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('email_value', $_POST['email'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['year']) || !is_numeric($_POST['year']) || !preg_match('/^\d+$/', $_POST['year'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('year_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
  else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('year_value', $_POST['year'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['limbs']) || !is_numeric($_POST['limbs']) ||($_POST['limbs']<1)||($_POST['limbs']>5)) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('limbs_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('limbs_value', $_POST['limbs'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['pol']) || !($_POST['pol']=='M' || $_POST['pol']=='W')) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('pol_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('pol_value', $_POST['pol'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['super']) || !is_array($_POST['super']) || (int)$_POST['super']<1 || (int)$_POST['super']>3) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('super_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('super_value', serialize($_POST['super']), time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['biography'])) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('biography_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
        // Сохраняем ранее введенное в форму значение на месяц.
        setcookie('biography_value', $_POST['biography'], time() + 30 * 24 * 60 * 60);
    }

    if (empty($_POST['check-1']) || !($_POST['check-1'] == 'on' || $_POST['check-1'] == 1)) {
        // Выдаем куку на день с флажком об ошибке в поле fio.
        setcookie('check_1_error', '1', time() + 24 * 60 * 60);
        $errors = TRUE;
    }
    else {
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
  }
  else {
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

  // Сохранение в БД.
$user = 'u52830';
$pass = '7841698';
//
$db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, [PDO::ATTR_PERSISTENT => true]);

// Подготовленный запрос. Не именованные метки.
//
 try {
     $stmt = $db->prepare("REPLACE INTO application SET name = ?,email = ?,year = ?,pol = ?,kol_kon = ?,biography = ?,ccheck = ?");
     $stmt -> execute([$_POST['fio'], $_POST['email'], (int)$_POST['year'], $_POST['pol'], (int)$_POST['limbs'], $_POST['biography'], 1]);
 }
 catch(PDOException $e){
     print('Error : ' . $e->getMessage());
     exit();
 }

$us_last = $db->lastInsertId();

foreach ($_POST['super'] as $super){
    try{
        $stmt = $db->prepare("REPLACE INTO userconnection SET idap = ?, idsuper = ?");
        $stmt->execute([$us_last, $super]);
    } catch (PDOException $e) {
        print('Error : ' . $e->getMessage());
        exit();
    }
}

  // Сохраняем куку с признаком успешного сохранения.
  setcookie('save', '1');

  // Делаем перенаправление.
  header('Location: index.php');
}
