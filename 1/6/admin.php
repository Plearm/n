<?php

/**
 * Задача 6. Реализовать вход администратора с использованием
 * HTTP-авторизации для просмотра и удаления результатов.
 **/

// Пример HTTP-аутентификации.
// PHP хранит логин и пароль в суперглобальном массиве $_SERVER.
// Подробнее см. стр. 26 и 99 в учебном пособии Веб-программирование и веб-сервисы.
if (empty($_SERVER['PHP_AUTH_USER']) ||
    empty($_SERVER['PHP_AUTH_PW'])){
    $user = 'u52830';
    $pass = '7841698';
    $db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, array(PDO::ATTR_PERSISTENT => true));

    $stmt = $db->prepare("SELECT login, password FROM admin WHERE login = ?");
    $stmt->execute([$_SERVER['PHP_AUTH_USER']]);
    $row = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($row) {
        $login= $row['login'];
        $password = $row['password'];
    } else {
        Login();
    }
    if ($row) {
        $login= $row['login'];
        $password = $row['password'];
    } else {
        Login();
    }
    
    if ($_SERVER['PHP_AUTH_USER'] != $login ||
    md5($_SERVER['PHP_AUTH_PW']) != $password){
        Login();
    }
    Login();
}
session_start();

function Login(){
    header('HTTP/1.1 401 Unanthorized');
    header('WWW-Authenticate: Basic realm="My site"');
    print('<h1>401 Неправильный логин или пароль</h1>');
    exit();
}
