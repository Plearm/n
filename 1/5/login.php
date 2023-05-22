<?php
    header('Content-Type: text/html; charset=UTF-8');
    <style>
        .error {
            border: 2px solid black;
            border-radius: 5px;
            background-color: black;
            color: white;
        }
    </style>

    session_start();

    if(!empty($_SESSION['login'])) {
        die('Вы уже авторизованы');
        header('Location: ./');
    }
    $user = get_user_by_login($_POST['login']);
    if (!$user) {
        die('Пользователь не найден');
    }
    if ($_POST['pass'] !== $user['pass']) {
        die('Неверный пароль');
    }
    $_SESSION['user_id'] = $user['id'];
    die('Привет, '.$user['login']);
    if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    ?>
        <form action="" method="post">
            <input name="login" />
            <input name="pass" />
            <input type="submit" value="Войти" />
        </form>    
        <?php
    }else {
        $sql = mysql_query("Select * from userlist where name = '$name'and pass='$pass';" );
        if (mysql_num_rows($sql) <= 0)   {
            $messages[] = '<div class="error">Неверно введен логин или пароль.</div>';
        }else {
            $_SESSION['login'] = $_POST['login'];
            $_SESSION['uid'] = 123;
            header('Location: ./');
        }
    }

?>
