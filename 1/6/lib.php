<?php 
function connectToDB($user,$pass){
    $db = new PDO('mysql:host=localhost;dbname=u47755', $user, $pass, array(PDO::ATTR_PERSISTENT => true));
    return $db;
}
function connectDB(){
    $user = 'u47755';
    $pass = '2914865';
    $db = new PDO('mysql:host=localhost;dbname=u47755', $user, $pass, array(PDO::ATTR_PERSISTENT => true));
    return $db;
}
?>
