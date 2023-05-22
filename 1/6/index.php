<?php

include('admin.php');

$user = 'u52830';
$pass = '7841698';
$db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, [PDO::ATTR_PERSISTENT => true]);

if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    print('Вы успешно авторизовались и видите защищенные паролем данные.');
    try {
        $stmt = $db->prepare("SELECT id, name, email, year, pol, kol_kon , biography FROM application");
        $stmt->execute();
        $values = $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (PDOException $e) {
        print('Error : ' . $e->getMessage());
        exit();
    }
    $message=array();
    $errors = array();

    $errors['error_id'] = empty($_COOKIE['error_id']) ? '' : $_COOKIE['error_id'];
    $errors['fio'] = !empty($_COOKIE['fio_error']);
    $errors['email'] = !empty($_COOKIE['email_error']);
    $errors['year'] = !empty($_COOKIE['year_error']);
    $errors['limbs'] = !empty($_COOKIE['limbs_error']);
    $errors['pol'] = !empty($_COOKIE['pol_error']);
    $errors['super'] = !empty($_COOKIE['super_error']);
    $errors['biography'] = !empty($_COOKIE['biography_error']);
    $errors['check-1'] = !empty($_COOKIE['check_1_error']);

    if (!empty($errors['fio'])) {
        setcookie('fio_error', '', 100000);
        $messages['fio'] = '<p class="msg">Не заполнено поле имени</p>';
    }
    if (!empty($errors['email'])) {
        setcookie('email_error', '', 100000);
        $messages['email'] = '<p class="msg">Не заполнено поле email</p>';
    }
    if (!empty($errors['year'])) {
        setcookie('year_error', '', 100000);
        $messages['year'] = '<p class="msg">Не заполнено поле возраста</p>';
    }
    if (!empty($errors['pol'])) {
        setcookie('pol_error', '', 100000);
        $messages['pol'] = '<p class="msg">Не выбран пол</p>';
    }
    if (!empty($errors['limbs'])) {
        setcookie('limbs_error', '', 100000);
        $messages['limbs'] = '<p class="msg">Не выбраны конечности</p>';
    }
    if (!empty($errors['super'])) {
        setcookie('super_error', '', 100000);
        $messages['super'] = '<p class="msg">Не выбрана ни одна сверхспособность</p>';
    }
    if (!empty($errors['biography'])) {
        setcookie('biography_error', '', 100000);
        $messages['biography'] = '<p class="msg">Не заполнено поле биографии</p>';
    }
    $_SESSION['token'] = bin2hex(random_bytes(32));
    include('admin_panel.php');
    exit();
}
else{
    if (!empty($_POST['token']) && hash_equals($_POST['token'], $_SESSION['token'])) {
    foreach ($_POST as $val => $value) {
        if (preg_match('/^clear(\d+)$/', $val, $matches)){
            $app_id = $matches[1];
            setcookie('clear', $app_id, time() + 24 * 60 * 60);
            $stmt = $db->prepare("DELETE FROM user WHERE app_id  = ?");
            $stmt->execute([$app_id]);
            $stmt = $db->prepare("DELETE FROM userconnection WHERE idap = ?");
            $stmt->execute([$app_id]);
            $stmt = $db->prepare("DELETE FROM application WHERE id = ?");
            $stmt->execute([$app_id]);
        }
        if(preg_match('/^save(\d+)$/', $val, $matches)){
            $app_id = $matches[1];
            $mas = array();
            $mas['name'] = $_POST['fio' . $app_id];
            $mas['email'] = $_POST['email' . $app_id];
            $mas['year'] = $_POST['year' . $app_id];
            $mas['pol'] = $_POST['pol' . $app_id];
            $mas['kol_kon'] = $_POST['limbs' . $app_id];
            $abilities = $_POST['super' . $app_id];
            $filtred_abilities = array_filter($abilities, function($value) {return($value == 1 || $value == 2 || $value == 3);});
            $mas['biography'] = $_POST['biography' . $app_id];
            $fio = $mas['name'];
            $email = $mas['email'];
            $year = $mas['year'];
            $pol = $mas['pol'];
            $limbs = $mas['kol_kon'];
            $biography = $mas['biography'];
            $errors = FALSE;

            if (empty($fio)) {
                setcookie('fio_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (empty($email)) {
                setcookie('email_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (!is_numeric($year) || (2023 - $year) < 14) {
                setcookie('year_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (empty($pol) || ($pol != 'M' && $pol != 'W')) {
                setcookie('pol_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (empty($limbs) || ($limbs<1 && $limbs>5)) {
                setcookie('limbs_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (empty($abilities) || count($filtred_abilities) != count($abilities)) {
                setcookie('super_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if (empty($biography)) {
                setcookie('biography_error', '1', time() + 24 * 60 * 60);
                $errors = TRUE;
            }
            if ($errors) {
                setcookie('error_id', $app_id, time() + 24 * 60 * 60);
                header('Location: index.php');
                exit();
            } else {
                setcookie('name_error', '', 100000);
                setcookie('email_error', '', 100000);
                setcookie('year_error', '', 100000);
                setcookie('pol_error', '', 100000);
                setcookie('limbs_error', '', 100000);
                setcookie('super_error', '', 100000);
                setcookie('biography_error', '', 100000);
                setcookie('error_id', '', 100000);
            }
            $stmt = $db->prepare("SELECT name, email, year, pol, kol_kon, biography FROM application WHERE id = ?");
            $stmt->execute([$app_id]);
            $old_dates = $stmt->fetchAll(PDO::FETCH_ASSOC);

            $stmt = $db->prepare("SELECT idsuper FROM userconnection WHERE idap = ?");
            $stmt->execute([$app_id]);
            $old_abilities = $stmt->fetchAll(PDO::FETCH_COLUMN);
            var_dump($mas);
            var_dump($old_dates[0]);
            if (array_diff($mas, $old_dates[0])) {
                $stmt = $db->prepare("UPDATE application SET name = ?, email = ?, year = ?, pol = ?, kol_kon = ?, biography = ? WHERE id = ?");
                $stmt->execute([$mas['name'], $mas['email'], $mas['year'], $mas['pol'], $mas['kol_kon'], $mas['biography'], $app_id]);
            }
            if (array_diff($abilities, $old_abilities) || count($abilities) != count($old_abilities)) {
                $stmt = $db->prepare("DELETE FROM userconnection WHERE idap = ?");
                $stmt->execute([$app_id]);
                $stmt = $db->prepare("INSERT INTO userconnection (idap, idsuper) VALUES (?, ?)");
                foreach ($abilities as $super_id) {
                    $stmt->execute([$app_id, $super_id]);
                }
            }
        }
    }
    header('Location: index.php');
    }
    else{
        var_dump($_POST['token']);
        var_dump($_SESSION['token']);
        die('Ошибка CSRF: недопустимый токен');
    }
}
